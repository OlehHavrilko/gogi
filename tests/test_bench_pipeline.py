"""Чистые функции бенчмарка (метрики). Секции STT/LLM/TTS требуют живого
железа и здесь не проверяются — только счётная часть."""

from scripts.bench_pipeline import _cer, _levenshtein


def test_levenshtein_identical():
    assert _levenshtein("привет", "привет") == 0


def test_levenshtein_counts_edits():
    assert _levenshtein("кот", "код") == 1
    assert _levenshtein("", "абв") == 3


def test_cer_perfect_match_is_zero():
    assert _cer("Открой калькулятор.", "открой калькулятор") == 0.0


def test_cer_ignores_case_trailing_punct_and_whitespace():
    assert _cer("  ОТКРОЙ   КАЛЬКУЛЯТОР !! ", "открой калькулятор") == 0.0


def test_cer_is_fraction_of_reference_length():
    # одна замена символа в референсе из 3 значимых символов -> 1/3
    assert abs(_cer("кот", "код") - 1 / 3) < 1e-9


def test_cer_empty_reference_is_zero():
    assert _cer("что-то", "") == 0.0
