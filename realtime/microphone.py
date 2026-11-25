"""Microphone capture. Used only by live_cli.py (the WebSocket path receives audio from the client)."""
import queue

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16_000


class Microphone:
    def __init__(self, device=None, block_seconds: float = 0.1):
        self._queue: queue.Queue = queue.Queue()
        self._running = False
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=int(SAMPLE_RATE * block_seconds),
            device=device,
            callback=self._on_audio,
        )

    def _on_audio(self, indata, frames, time_info, status):
        # Runs on PortAudio's audio thread: keep it tiny. Just copy and queue.
        if status:
            print(f"[mic warning] {status}")
        self._queue.put(indata[:, 0].copy())

    def __enter__(self):
        self._running = True
        self._stream.start()
        return self

    def __exit__(self, *exc):
        self._running = False
        self._stream.stop()
        self._stream.close()

    def blocks(self):
        """Yield audio blocks until the microphone is closed."""
        # get() uses a timeout so Ctrl+C can interrupt it. A plain blocking
        # get() often can't be interrupted on Windows.
        while self._running:
            try:
                yield self._queue.get(timeout=0.2)
            except queue.Empty:
                continue