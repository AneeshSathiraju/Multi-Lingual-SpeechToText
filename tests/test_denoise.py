import shutil

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routes import router
from audio.noise_filter import reduce_noise
from config import Settings

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg not on PATH")
SR = 16_000


def _tone(seconds=3):
    t = np.arange(seconds * SR) / SR
    return (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def test_reduces_steady_noise_on_synthetic_signal():
    clean = _tone()
    noisy = clean + np.random.default_rng(0).normal(0, 0.05, clean.size).astype(np.float32)
    cleaned = reduce_noise(noisy, SR, strength=1.0)

    err_before = np.mean((noisy - clean) ** 2)
    err_after = np.mean((cleaned - clean) ** 2)
    assert err_after < err_before  # synthetic only; says nothing about real speech


def test_output_keeps_length_and_dtype():
    out = reduce_noise(_tone(1), SR)
    assert out.dtype == np.float32 and out.size == SR


def test_empty_audio_rejected():
    with pytest.raises(ValueError):
        reduce_noise(np.array([], dtype=np.float32), SR)


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(router)
    app.state.settings = Settings()
    return TestClient(app)


@needs_ffmpeg
def test_endpoint_returns_wav(client, wav_bytes):
    r = client.post("/api/v1/denoise", files={"file": ("t.wav", wav_bytes, "audio/wav")})
    assert r.status_code == 200
    assert r.headers["content-type"] == "audio/wav"
    assert r.content[:4] == b"RIFF"


def test_endpoint_rejects_bad_strength(client, wav_bytes):
    r = client.post("/api/v1/denoise?strength=5", files={"file": ("t.wav", wav_bytes, "audio/wav")})
    assert r.status_code == 422