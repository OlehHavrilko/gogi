"""StreamingSpeaker: сборка токенов в куски для TTS. Первый кусок ответа
режется по раннему рубежу (запятая/предел длины), дальше — по предложениям."""

from tts_engine import StreamingSpeaker


class _RecordingEngine:
    def __init__(self):
        self.said: list[str] = []

    def say(self, text):
        self.said.append(text)


def _feed(speaker, text, *, step=3):
    for i in range(0, len(text), step):
        speaker.feed(text[i : i + step])


def test_first_chunk_emitted_on_comma_not_waiting_for_period():
    eng = _RecordingEngine()
    sp = StreamingSpeaker(eng)
    _feed(sp, "Секунду, сейчас посмотрю и всё сделаю.")
    sp.flush()
    # первым ушёл кусок до запятой, а не всё предложение целиком
    assert eng.said[0] == "Секунду,"
    assert "сейчас посмотрю и всё сделаю." in eng.said[-1]


def test_subsequent_chunks_split_on_sentences_only():
    eng = _RecordingEngine()
    sp = StreamingSpeaker(eng)
    _feed(sp, "Готово. Дальше, если нужно, покажу ещё. Спрашивай.")
    sp.flush()
    # после первого куска запятая внутри «если нужно» не должна резать
    assert "Дальше, если нужно, покажу ещё." in eng.said


def test_first_chunk_hard_cap_when_no_punctuation():
    eng = _RecordingEngine()
    sp = StreamingSpeaker(eng)
    long_head = "слово " * 20  # ни точки, ни запятой, длиннее предела
    _feed(sp, long_head)
    assert eng.said, "должен сработать аварийный предел по длине"
    assert len(eng.said[0]) <= 60


def test_flush_emits_remainder_and_resets_first_chunk_flag():
    eng = _RecordingEngine()
    sp = StreamingSpeaker(eng)
    _feed(sp, "Первый ответ, с продолжением.")
    sp.flush()
    said_after_turn1 = list(eng.said)

    # новый ход — снова должен работать ранний рубеж по запятой
    eng.said.clear()
    _feed(sp, "Второй ответ, тоже с запятой.")
    assert eng.said[0] == "Второй ответ,"
    assert said_after_turn1  # первый ход что-то произнёс


def test_reset_drops_buffer_without_emitting():
    eng = _RecordingEngine()
    sp = StreamingSpeaker(eng)
    _feed(sp, "недоговорённый хвост без знаков")
    before = list(eng.said)
    sp.reset()
    sp.feed("!")  # ничего от старого буфера долететь не должно
    assert eng.said == before
