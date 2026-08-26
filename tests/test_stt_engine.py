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


def test_stop_recording_without_start_returns_empty_array():
    engine = stt_engine.STTEngine.__new__(stt_engine.STTEngine)
    engine._stream = None
    result = engine.stop_recording()
    assert result.size == 0


def test_start_stop_recording_concatenates_frames(mocker):
    fake_stream = mocker.Mock()
    mocker.patch.object(stt_engine.sd, "InputStream", return_value=fake_stream)

    engine = stt_engine.STTEngine.__new__(stt_engine.STTEngine)
    engine.start_recording()

    # имитируем то, что делает callback InputStream во время записи
    callback = stt_engine.sd.InputStream.call_args.kwargs["callback"]
    callback(np.ones((10, 1), dtype=np.float32), 10, None, None)
    callback(np.zeros((5, 1), dtype=np.float32), 5, None, None)

    fake_stream.start.assert_called_once()

    audio = engine.stop_recording()

    assert audio.shape == (15,)
    fake_stream.stop.assert_called_once()
    fake_stream.close.assert_called_once()
    assert engine._stream is None
