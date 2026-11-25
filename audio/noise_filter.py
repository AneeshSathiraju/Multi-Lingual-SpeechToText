"""Noise reduction by spectral gating (noisereduce).

Works best on steady noise (fans, hum, hiss). It cannot remove other voices,
and strong settings can leave artifacts or muffle the speech.
"""
import numpy as np
import noisereduce as nr


def reduce_noise(
    audio: np.ndarray,
    sample_rate: int,
    strength: float = 0.8,
    stationary: bool = True,
) -> np.ndarray:
    """
    audio:      mono float32 samples.
    strength:   0.0 (no change) to 1.0 (remove as much as possible).
    stationary: True  = assume constant noise; estimated from the whole clip (faster).
                False = adapt over time; handles changing noise better but is slower
                        and more likely to touch the speech.
    """
    if audio.size == 0:
        raise ValueError("No audio to process.")
    cleaned = nr.reduce_noise(
        y=audio,
        sr=sample_rate,
        stationary=stationary,
        prop_decrease=strength,
    )
    return cleaned.astype(np.float32)