import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.ws import router as ws_router
from config import Settings
from realtime.chunker import UtteranceChunker
from realtime.stream import LiveTranscriber
from recognition.transcriber import TranscriptionResult

BLOCK = 1600  # 0.1 s at 16 kHz


def loud():
    t = np.arange(BLOCK) / 16000
    return (0.3 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)


def quiet():
    return np.zeros(BLOCK, dtype=np.float32)


class FakeTranscriber:
    model_size, device, compute_type = "fake", "cpu", "int8"

    def transcribe(self, audio, language=None, beam_size=5):
        return TranscriptionResult("hello", language or "en", None, [])


def test_speech_then_silence_yields_one_utterance():
    c = UtteranceChunker(threshold=0.01)
    results = [c.feed(b) for b in [quiet()] * 3 + [loud()] * 10 + [quiet()] * 10]
    utterances = [r for r in results if r is not None]
    assert len(utterances) == 1
    assert len(utterances[0]) >= 10 * BLOCK  # includes the speech


def test_silence_alone_yields_nothing():
    c = UtteranceChunker(threshold=0.01)
    assert all(c.feed(quiet()) is None for _ in range(50))


def test_short_blip_is_discarded():
    c = UtteranceChunker(threshold=0.01)
    results = [c.feed(b) for b in [loud()] + [quiet()] * 10]
    assert all(r is None for r in results)


def test_live_transcriber_emits_final():
    live = LiveTranscriber(FakeTranscriber(), language="en", chunker=UtteranceChunker(threshold=0.01))
    events = []
    for b in [loud()] * 5 + [quiet()] * 10:
        events += live.feed(b)
    assert [e.type for e in events] == ["final"]
    assert events[0].text == "hello"


def test_websocket_flush_returns_final():
    app = FastAPI()
    app.include_router(ws_router)
    app.state.settings = Settings()
    app.state.transcriber = FakeTranscriber()
    pcm = (loud() * 32767).astype("<i2").tobytes()

    with TestClient(app).websocket_connect("/api/v1/live?language=hi") as ws:
        assert ws.receive_json()["type"] == "ready"
        for _ in range(5):
            ws.send_bytes(pcm)
        ws.send_text("end")
        event = ws.receive_json()
        assert event["type"] == "final"
        assert event["language"] == "hi"