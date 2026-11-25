"""Glue: chunker + transcriber -> partial/final events. No FastAPI, no microphone."""
from dataclasses import dataclass

from realtime.chunker import UtteranceChunker, SAMPLE_RATE


@dataclass
class LiveEvent:
    type: str        # "partial" or "final"
    text: str
    language: str
    utterance: int   # which utterance this belongs to (a partial is replaced by its final)


class LiveTranscriber:
    """One instance per stream. The Transcriber (and its model) is shared."""

    def __init__(self, transcriber, language=None, chunker=None, partial_interval=1.5):
        self._t = transcriber
        self._language = language
        self._chunker = chunker or UtteranceChunker()
        self._partial_samples = int(partial_interval * SAMPLE_RATE)
        self._since_partial = 0
        self._utterance = 0

    def feed(self, block) -> list[LiveEvent]:
        utterance = self._chunker.feed(block)
        if utterance is not None:
            return self._finalize(utterance)

        if not self._chunker.in_speech:
            self._since_partial = 0
            return []

        self._since_partial += len(block)
        if self._since_partial < self._partial_samples:
            return []
        self._since_partial = 0
        # beam_size=1 (greedy) keeps partials cheap
        result = self._t.transcribe(self._chunker.current_audio(), self._language, beam_size=1)
        if not result.text:
            return []
        return [LiveEvent("partial", result.text, result.language, self._utterance)]

    def flush(self) -> list[LiveEvent]:
        utterance = self._chunker.flush()
        return self._finalize(utterance) if utterance is not None else []

    def _finalize(self, audio) -> list[LiveEvent]:
        self._since_partial = 0
        result = self._t.transcribe(audio, self._language, beam_size=5)
        index = self._utterance
        self._utterance += 1
        if not result.text:
            return []
        return [LiveEvent("final", result.text, result.language, index)]