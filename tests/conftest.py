import io
import wave

import numpy as np
import pytest


@pytest.fixture
def wav_bytes() -> bytes:
    """One second of a 440 Hz tone as a valid WAV file."""
    sr = 16_000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    samples = (0.3 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(samples.tobytes())
    return buf.getvalue()