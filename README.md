# Домовой — локальный голосовой ассистент

Полностью локальный voice-ассистент в стиле Jarvis: распознавание речи, LLM с
function calling и синтез речи с клонированным голосом — всё работает на этой
машине, без обращений в облако.

## Стек
- **STT:** faster-whisper (`medium`, CPU, int8)
- **LLM:** gpt-oss:20b через Ollama, streaming + tool calling, на GPU
- **TTS:** F5-TTS-Russian (дообучен на русском), voice cloning по референсу,
  GPU через ROCm

## Железо и GPU-ускорение
Стенд собран на **AMD Radeon RX 9060 XT** без CUDA. Чтобы PyTorch увидел GPU
на Windows, используется preview-сборка **ROCm 7.2.1**:

```bash
pip install --no-cache-dir \
  https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_core-7.2.1-py3-none-win_amd64.whl \
  https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_devel-7.2.1-py3-none-win_amd64.whl \
  https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_libraries_custom-7.2.1-py3-none-win_amd64.whl \
  https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm-7.2.1.tar.gz

pip install --no-cache-dir \
  https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torch-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl \
  https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torchaudio-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl \
  https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torchvision-0.24.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl
```

Требуется свежий графический драйвер AMD (Adrenalin ≥ 26.2.2 — на практике
подходит практически любой актуальный драйвер 2026 года).

**Важно:** любой пакет, который тянет `torch` как зависимость (например,
`chatterbox-tts`), может незаметно перезатереть ROCm-сборку обычным CPU-torch
из PyPI. После установки новых пакетов всегда проверяйте:
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```
Если вернулась CPU-версия — переустановите ROCm-колёса с `--force-reinstall --no-deps`.

## Известные баги окружения и их обходы (уже зашиты в `tts_engine.py`)
1. **`torch.distributed` отсутствует** в ROCm-сборке для Windows — `encodec`
   (вокодер внутри F5-TTS) падает на `ReduceOp` при импорте. Патчится заглушкой.
2. **`torchaudio.load()` требует `torchcodec`**, а его `.dll` собран под другой
   ABI torch и не грузится. Аудио читаем напрямую через `soundfile`.
3. **Конфликт OpenMP-рантаймов** между CTranslate2 (whisper) и ROCm-torch —
   нужен `KMP_DUPLICATE_LIB_OK=TRUE`.
4. **`pkg_resources` вырезан** из новых версий `setuptools` — если ставите
   `chatterbox-tts` или что-то ещё, зависящее от `perth`/`pkg_resources`,
   нужен `pip install "setuptools<81"`.
5. **`TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`** — включает более быстрые
   ядра внимания на AMD, даёт ~2x на F5-TTS.
6. Чекпоинт `hotstone228/F5-TTS-Russian` обучен на архитектуре
   **`F5TTS_Base`**, а не `F5TTS_v1_Base` (дефолт библиотеки `f5-tts`) — с
   неправильным конфигом модель генерирует нечленораздельный шум вместо речи.

## Голос
Референс для voice cloning — `voices/gogi_voice.wav` (короткий сэмпл
персонажа с кавказским акцентом) + `voices/gogi_transcript.txt` (точный
транскрипт, распознанный нашим же STT). Чтобы сменить голос — замените оба
файла на новый референс (5-15 секунд чистой речи) и его точный транскрипт.

Параметры синтеза (`tts_engine.py`): `nfe_step=32`, `cfg_strength=3.0` —
баланс скорости (~7-10с на фразу) и выразительности, подобранный вручную.

## Первый запуск
```bash
ollama serve          # в отдельном терминале, если ещё не запущен
ollama pull gpt-oss:20b
```

Скачать веса F5-TTS-Russian (не хранятся в репо, ~1.35 ГБ):
```bash
curl -L -o voices/f5_ru/model_last.safetensors \
  https://huggingface.co/hotstone228/F5-TTS-Russian/resolve/main/model_last.safetensors
curl -L -o voices/f5_ru/vocab.txt \
  https://huggingface.co/hotstone228/F5-TTS-Russian/resolve/main/vocab.txt
```

```bash
venv\Scripts\python main.py
```

## Как пользоваться
- Нажми Enter — начинается запись с микрофона.
- Говори.
- Нажми Enter ещё раз — запись останавливается, идёт распознавание и ответ.
- Ctrl+C — выход.

## Открытие приложений
Список разрешённых приложений — в `tools.py` (`APPS`), плюс словарь русских
алиасов (`ALIASES`), т.к. модель иногда называет приложение по-русски вопреки
enum. Это осознанное ограничение: LLM может запускать только то, что явно в
белом списке, а не произвольные команды.

## Известные ограничения прототипа
- Push-to-talk (Enter/Enter), а не voice activity detection.
- Латентность синтеза (~7-10с на фразу) упирается в незрелые ROCm-ядра
  (MIOpen на этой GPU-архитектуре ещё не оптимизирован под конв-тяжёлые
  модели вроде XTTS — там 30-65с; F5-TTS справляется лучше, т.к. в основном
  attention-based).
- Нет UI для выбора голоса — следующий шаг развития проекта.
