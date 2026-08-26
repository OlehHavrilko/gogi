# Гоги — локальный голосовой ассистент

Полностью локальный voice-ассистент в стиле Jarvis: распознавание речи, LLM с
function calling и синтез речи с клонированным голосом — всё работает на этой
машине, без обращений в облако.

## Стек
- **STT:** faster-whisper (`medium`, CPU, int8)
- **LLM:** gpt-oss:20b через Ollama, streaming + tool calling, на GPU
- **TTS:** F5-TTS-Russian (дообучен на русском), voice cloning по референсу,
  GPU через ROCm

## Железо и GPU-ускорение
Поддерживаются **NVIDIA (CUDA)** и **AMD (ROCm)** — код сам определяет
бэкенд в рантайме (`gpu.py`, `torch.cuda.is_available()` возвращает `True`
на обеих платформах, HIP мимикрирует CUDA-неймспейс). Никаких правок кода
не требуется — нужно только правильно поставить сам `torch` под свою карту.
Без GPU всё тоже работает, просто на CPU (медленно для TTS).

### NVIDIA (CUDA)
Обычные колёса с PyTorch-репозитория, версия CUDA — под ваш драйвер
(см. https://pytorch.org/get-started/locally/, на практике подходит любая
свежая ветка cu12x):

```bash
pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu124
```

### AMD (ROCm, Windows)
Стенд для разработки собран на **AMD Radeon RX 9060 XT**. Чтобы PyTorch
увидел GPU на Windows, используется preview-сборка **ROCm 7.2.1**:

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

Проверка после установки (одинаково для NVIDIA и AMD):
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```
При старте `main.py`/`test_pipeline.py` также печатают, какой бэкенд
обнаружен (`[TTS] GPU-бэкенд: NVIDIA CUDA 12.4` / `AMD ROCm (HIP ...)` / `CPU`).

**Важно:** любой пакет, который тянет `torch` как зависимость (например,
`chatterbox-tts`), может незаметно перезатереть ROCm-сборку обычным CPU-torch
из PyPI. После установки новых пакетов всегда проверяйте:
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```
Если вернулась CPU-версия — переустановите ROCm-колёса с `--force-reinstall --no-deps`.

## Известные баги окружения и их обходы (уже зашиты в `tts_engine.py`)
Пункты 1, 3 и 5 специфичны для ROCm-сборки и на NVIDIA/CUDA обычно не
проявляются вовсе (или срабатывают безвредно) — код в `gpu.py`/`tts_engine.py`
проверяет условия перед патчем, поэтому ничего не нужно включать/выключать
вручную под конкретный вендор.

1. **`torch.distributed` отсутствует** в ROCm-сборке для Windows — `encodec`
   (вокодер внутри F5-TTS) падает на `ReduceOp` при импорте. Патчится заглушкой.
2. **`torchaudio.load()` в некоторых сборках требует `torchcodec`**, а его
   `.dll` собран под другой ABI torch и не грузится. Аудио читаем напрямую
   через `soundfile` всегда — это безопасно независимо от вендора GPU.
3. **Конфликт OpenMP-рантаймов** между CTranslate2 (whisper) и GPU-torch —
   нужен `KMP_DUPLICATE_LIB_OK=TRUE`.
4. **`pkg_resources` вырезан** из новых версий `setuptools` — если ставите
   что-то, зависящее от `perth`/`pkg_resources`, нужен
   `pip install "setuptools<81"`.
5. **`TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`** — включает более быстрые
   ядра внимания на AMD, даёт ~2x на F5-TTS. На NVIDIA игнорируется.
6. Чекпоинт `hotstone228/F5-TTS-Russian` обучен на архитектуре
   **`F5TTS_Base`**, а не `F5TTS_v1_Base` (дефолт библиотеки `f5-tts`) — с
   неправильным конфигом модель генерирует нечленораздельный шум вместо речи.

## Голос
Референс для voice cloning — `voices/gogi_voice.wav` (короткий сэмпл
персонажа с кавказским акцентом) + `voices/gogi_transcript.txt` (точный
транскрипт, распознанный нашим же STT). Чтобы сменить голос — замените оба
файла на новый референс (5-15 секунд чистой речи) и его точный транскрипт.

Параметры синтеза настраиваются в конфиге (`tts.nfe_step`, `tts.cfg_strength`,
дефолт 32/3.0) — баланс скорости (~7-10с на фразу) и выразительности,
подобранный вручную.

## Конфигурация
Все настройки (модель LLM, системный промпт, параметры STT/TTS, список
разрешённых приложений и их пути) — в `config.yaml`. Файл не хранится в
репозитории (там пользовательские пути), дефолты лежат в
`config.example.yaml`. Перед первым запуском:

```bash
cp config.example.yaml config.yaml
```

и поправьте пути приложений под свою систему (значения `%APPDATA%` и т.п.
разворачиваются автоматически, менять на абсолютные пути не обязательно).
Если `config.yaml` отсутствует, используются дефолты из
`config.example.yaml` напрямую.

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
Список разрешённых приложений — в `config.yaml` (`apps`), плюс словарь русских
алиасов (`aliases`), т.к. модель иногда называет приложение по-русски вопреки
enum. Это осознанное ограничение: LLM может запускать только то, что явно в
белом списке, а не произвольные команды.

## Тесты
Автотесты (`tests/`) не требуют GPU, микрофона или скачанных весов моделей —
все тяжёлые классы (`WhisperModel`, `F5TTS`, Ollama) мокаются:

```bash
pip install -r requirements-dev.txt
pytest
ruff check .
```

Скрипты в `scripts/` — не автотесты, а ручные проверки на живом железе
(запускаются вручную, требуют реальный GPU/Ollama/референс-аудио).

## Известные ограничения прототипа
- Push-to-talk (Enter/Enter), а не voice activity detection.
- Латентность синтеза (~7-10с на фразу) упирается в незрелые ROCm-ядра
  (MIOpen на этой GPU-архитектуре ещё не оптимизирован под конв-тяжёлые
  модели вроде XTTS — там 30-65с; F5-TTS справляется лучше, т.к. в основном
  attention-based).
- Нет UI для выбора голоса — следующий шаг развития проекта.
