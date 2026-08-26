"""Проверка STTEngine без загрузки реальной модели whisper."""

import numpy as np

import stt_engine


def test_transcribe_empty_audio_returns_empty_string(mocker):
    mocker.patch.object(stt_engine, "WhisperModel")
    engine = stt_engine.STTEngine.__new__(stt_engine.STTEngine)  # без реальной загрузки модели
    result = engine.transcribe(np.zeros(0, dtype=np.float32))
    assert result == ""


def test_transcribe_joins_segments(mocker):
    fake_segment_1 = mocker.Mock(text=" привет ")
    fake_segment_2 = mocker.Mock(text=" мир ")
    mock_model = mocker.Mock()
    mock_model.transcribe.return_value = ([fake_segment_1, fake_segment_2], None)

    engine = stt_engine.STTEngine.__new__(stt_engine.STTEngine)
    engine.model = mock_model

    result = engine.transcribe(np.ones(1600, dtype=np.float32))
    assert result == "привет мир"
    mock_model.transcribe.assert_called_once()
