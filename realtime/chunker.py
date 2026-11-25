"""Decide where speech starts and ends. Pure numpy: no model, no mic, no FastAPI."""
import collections

import numpy as np

SAMPLE_RATE = 16_000


def rms(block: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(block)))) if block.size else 0.0


class UtteranceChunker:
    """Energy-based voice activity detection.

    Limitation: it treats *any* loud sound as speech (a cough, a door slam,
    background chatter). A model-based VAD such as Silero would be more robust;
    that's a good upgrade later.
    """

    def __init__(self, threshold=0.01, end_silence=0.7, min_speech=0.3, max_utterance=15.0, preroll=0.3):
        self.threshold = threshold                      # RMS level that counts as "loud"
        self._end_silence = int(end_silence * SAMPLE_RATE)
        self._min_speech = int(min_speech * SAMPLE_RATE)
        self._max_length = int(max_utterance * SAMPLE_RATE)
        self._preroll_max = int(preroll * SAMPLE_RATE)  # keep a little audio from *before* speech starts
        self._preroll: collections.deque = collections.deque()
        self._preroll_len = 0
        self._reset()

    def _reset(self):
        self._blocks: list[np.ndarray] = []
        self._length = 0
        self._speech = 0     # samples that were loud
        self._silence = 0    # consecutive quiet samples at the end
        self._in_speech = False

    @property
    def in_speech(self) -> bool:
        return self._in_speech

    def current_audio(self) -> np.ndarray | None:
        """The utterance so far (used for partial results)."""
        return np.concatenate(self._blocks) if self._in_speech else None

    def feed(self, block: np.ndarray) -> np.ndarray | None:
        """Give it one block. Returns a finished utterance, or None."""
        loud = rms(block) >= self.threshold

        if not self._in_speech:
            if not loud:
                self._remember(block)
                return None
            # Speech just started: begin with the preroll so the first syllable isn't clipped.
            self._in_speech = True
            self._blocks = list(self._preroll)
            self._length = self._preroll_len
            self._preroll.clear()
            self._preroll_len = 0

        self._blocks.append(block)
        self._length += len(block)
        if loud:
            self._speech += len(block)
            self._silence = 0
        else:
            self._silence += len(block)

        # max length forces a cut; it can split a word, a known limitation
        if self._silence >= self._end_silence or self._length >= self._max_length:
            return self._finish()
        return None

    def flush(self) -> np.ndarray | None:
        """Call when the stream ends, to get any half-finished utterance."""
        return self._finish() if self._in_speech else None

    def _finish(self) -> np.ndarray | None:
        audio = np.concatenate(self._blocks)
        long_enough = self._speech >= self._min_speech  # drops clicks and coughs
        self._reset()
        return audio if long_enough else None

    def _remember(self, block: np.ndarray):
        self._preroll.append(block)
        self._preroll_len += len(block)
        while self._preroll_len > self._preroll_max and self._preroll:
            self._preroll_len -= len(self._preroll.popleft())