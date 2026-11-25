import io
import shutil

import pytest

from audio.loader import (
    AudioDecodeError, EmptyAudioError, FileTooLargeError, UnsupportedFormatError, temp_audio_file,
)
from audio.preprocessor import decode_to_mono_16k, duration_seconds

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg not on PATH")


def test_empty_file_rejected():
    with pytest.raises(EmptyAudioError):
        with temp_audio_file(io.BytesIO(b""), "a.wav", 1024):
            pass


def test_unsupported_extension_rejected():
    with pytest.raises(UnsupportedFormatError):
        with temp_audio_file(io.BytesIO(b"data"), "notes.txt", 1024):
            pass


def test_oversized_file_rejected():
    with pytest.raises(FileTooLargeError):
        with temp_audio_file(io.BytesIO(b"x" * 100), "a.wav", 10):
            pass


def test_temp_file_is_deleted_after_use():
    with temp_audio_file(io.BytesIO(b"abc"), "a.wav", 1024) as path:
        assert path.exists()
    assert not path.exists()


@needs_ffmpeg
def test_decode_gives_16k_float32_and_correct_duration(wav_bytes):
    with temp_audio_file(io.BytesIO(wav_bytes), "tone.wav", 10_000_000) as path:
        audio = decode_to_mono_16k(path)
    assert audio.dtype.name == "float32"
    assert duration_seconds(audio) == pytest.approx(1.0, abs=0.05)


@needs_ffmpeg
def test_garbage_bytes_fail_to_decode():
    with temp_audio_file(io.BytesIO(b"this is not audio"), "fake.wav", 1024) as path:
        with pytest.raises(AudioDecodeError):
            decode_to_mono_16k(path)