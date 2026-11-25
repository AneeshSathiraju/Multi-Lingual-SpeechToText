import os
import shutil

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routes import router
from config import Settings
from recognition.transcriber import Segment, TranscriptionResult

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg not on PATH")


class FakeTranscriber:
    """Stands in for the real model so API tests are fast and need no download."""
    model_size, device, compute_type = "fake", "cpu", "int8"

    def transcribe(self, audio, language=None):
        return TranscriptionResult(
            text="hello world",
            language=language or "en",
            language_probability=None if language else 0.99,
            segments=[Segment(0.0, 1.0, "hello world")],
        )


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(router)
    app.state.settings = Settings()
    app.state.transcriber = FakeTranscriber()
    return TestClient(app)  # no `with`, so no lifespan, so no model load


def test_health(client):
    assert client.get("/api/v1/health").json()["status"] == "ok"


def test_languages_lists_hindi_and_telugu(client):
    codes = {l["code"] for l in client.get("/api/v1/languages").json()["languages"]}
    assert {"en", "hi", "te"} <= codes


@needs_ffmpeg
def test_transcribe_auto_detect(client, wav_bytes):
    r = client.post("/api/v1/transcribe", files={"file": ("t.wav", wav_bytes, "audio/wav")})
    body = r.json()
    assert r.status_code == 200
    assert body["transcript"] == "hello world"
    assert body["language_source"] == "auto"
    assert body["audio_duration_seconds"] == pytest.approx(1.0, abs=0.05)


@needs_ffmpeg
def test_transcribe_explicit_language(client, wav_bytes):
    r = client.post("/api/v1/transcribe?language=hi", files={"file": ("t.wav", wav_bytes, "audio/wav")})
    body = r.json()
    assert body["language"] == "hi"
    assert body["language_source"] == "explicit"
    assert body["language_probability"] is None


def test_empty_upload_is_400(client):
    r = client.post("/api/v1/transcribe", files={"file": ("t.wav", b"", "audio/wav")})
    assert r.status_code == 400


def test_wrong_extension_is_415(client):
    r = client.post("/api/v1/transcribe", files={"file": ("t.txt", b"hello", "text/plain")})
    assert r.status_code == 415


def test_unknown_language_is_422(client, wav_bytes):
    r = client.post("/api/v1/transcribe?language=xx", files={"file": ("t.wav", wav_bytes, "audio/wav")})
    assert r.status_code == 422


@pytest.mark.skipif(not os.getenv("STT_RUN_MODEL_TESTS"), reason="downloads a model; set STT_RUN_MODEL_TESTS=1")
def test_real_engine_smoke():
    """Checks the real model loads and returns a result. Says nothing about accuracy."""
    from recognition.transcriber import Transcriber
    t = Transcriber("tiny", device="cpu")
    result = t.transcribe(np.zeros(16_000, dtype=np.float32), language="en")
    assert isinstance(result.text, str)
    assert result.language == "en"