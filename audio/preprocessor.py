"""Turn any audio file into what Whisper expects: 16 kHz, mono, float32 samples."""
import subprocess
from pathlib import Path

import numpy as np

from audio.loader import AudioDecodeError, EmptyAudioError, FFmpegNotFoundError

SAMPLE_RATE = 16_000
_FFMPEG_TIMEOUT_SECONDS = 300


def decode_to_mono_16k(path: Path) -> np.ndarray:
    cmd = [
        "ffmpeg", "-nostdin", "-v", "error",
        "-i", str(path),
        "-vn",                    # ignore any video stream
        "-ac", "1",               # mono
        "-ar", str(SAMPLE_RATE),  # resample to 16 kHz
        "-f", "f32le",            # raw 32-bit float samples
        "pipe:1",                 # write to stdout instead of a file
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


def duration_seconds(audio: np.ndarray) -> float:
    return audio.size / SAMPLE_RATE