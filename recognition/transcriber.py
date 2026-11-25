"""Use a loaded model to turn audio into text. Independent of FastAPI."""
import threading
from dataclasses import dataclass

import numpy as np

from recognition.engine import load_model
from recognition.language import normalize_language


@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class TranscriptionResult:
    text: str
    language: str
    language_probability: float | None  # None when the language was chosen explicitly
    segments: list[Segment]


class Transcriber:
    def __init__(self, model_size: str = "small", device: str = "auto", compute_type: str = "auto"):
        self.model_size = model_size
        self.model, self.device, self.compute_type = load_model(model_size, device, compute_type)
        # One model instance, possibly several simultaneous requests: take turns.
        self._lock = threading.Lock()

    def transcribe(self, audio: np.ndarray, language: str | None = None) -> TranscriptionResult:
        """audio: mono float32 at 16 kHz. language: a code like 'hi', or None to auto-detect."""
        language = normalize_language(language)

        with self._lock:
            segment_iter, info = self.model.transcribe(
                audio,
                language=language,   # None -> Whisper detects the language
                beam_size=5,
                vad_filter=True,     # skip silence; reduces made-up text on quiet stretches
            )
            # IMPORTANT: segment_iter is a generator. The real work happens as we
            # iterate, so it must be consumed inside the lock.
            segments = [Segment(s.start, s.end, s.text.strip()) for s in segment_iter]

        return TranscriptionResult(
            text=" ".join(s.text for s in segments).strip(),
            language=info.language,
            language_probability=None if language else info.language_probability,
            segments=segments,
        )
    def transcribe(self, audio: np.ndarray, language: str | None = None, beam_size: int = 5) -> TranscriptionResult:
        """audio: mono float32 at 16 kHz. language: a code like 'hi', or None to auto-detect."""
        language = normalize_language(language)

        with self._lock:
            segment_iter, info = self.model.transcribe(
                audio,
                language=language,   # None -> Whisper detects the language
                beam_size=beam_size,
                vad_filter=True,     # skip silence; reduces made-up text on quiet stretches
            )
            # segment_iter is a generator: consume it inside the lock.
            segments = [Segment(s.start, s.end, s.text.strip()) for s in segment_iter]

        return TranscriptionResult(
            text=" ".join(s.text for s in segments).strip(),
            language=info.language,
            language_probability=None if language else info.language_probability,
            segments=segments,
        )