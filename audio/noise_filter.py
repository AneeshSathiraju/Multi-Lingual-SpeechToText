"""Noise reduction. NOT IMPLEMENTED in V1.

Nothing in the pipeline calls this module, and transcripts are produced from
unfiltered audio. Planned for a later version.
"""
import numpy as np


def reduce_noise(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    raise NotImplementedError("Noise reduction is planned for a later version.")