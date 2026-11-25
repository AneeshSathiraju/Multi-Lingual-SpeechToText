"""Turn any audio file into what Whisper expects: 16 kHz, mono, float32 samples."""
import subprocess
from pathlib import Path

import numpy as np

from audio.loader import AudioDecodeError, EmptyAudioError, FFmpegNotFoundError

SAMPLE_RATE = 16_000
_FFMPEG_TIMEOUT_SECONDS = 300


import io
import wave

# ... keep the existing imports, SAMPLE_RATE and _FFMPEG_TIMEOUT_SECONDS ...


def decode_to_mono(path: Path, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    cmd = [
        "ffmpeg", "-nostdin", "-v", "error",
        "-i", str(path),
        "-vn",
        "-ac", "1",
        "-ar", str(sample_rate),
        "-f", "f32le",
        "pipe:1",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=_FFMPEG_TIMEOUT_SECONDS)
    except FileNotFoundError as exc:
        raise FFmpegNotFoundError("FFmpeg is not installed or not on PATH.") from exc
    except subprocess.TimeoutExpired as exc:
        raise AudioDecodeError("Decoding the audio timed out.") from exc

    if proc.returncode != 0:
        detail = proc.stderr.decode(errors="replace").strip()[-300:]
        raise AudioDecodeError(f"Could not decode audio: {detail}")

    audio = np.frombuffer(proc.stdout, dtype=np.float32)
    if audio.size == 0:
        raise EmptyAudioError("The file contains no audio samples.")
    return audio


def decode_to_mono_16k(path: Path) -> np.ndarray:
    """Existing callers keep working unchanged."""
    return decode_to_mono(path, SAMPLE_RATE)


def duration_seconds(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> float:
    return audio.size / sample_rate


def encode_wav(audio: np.ndarray, sample_rate: int) -> bytes:
    """Mono float32 in [-1, 1] -> 16-bit PCM WAV bytes (stdlib only)."""
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()

def duration_seconds(audio: np.ndarray) -> float:
    return audio.size / SAMPLE_RATE