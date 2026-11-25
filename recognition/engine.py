"""Model loading only. No FastAPI, no audio files."""
import importlib
import logging
import os
import sys

import numpy as np

logger = logging.getLogger(__name__)


def _register_nvidia_pip_dlls() -> None:
    """Windows: let CTranslate2 find cuBLAS/cuDNN installed via pip
    (nvidia-cublas-cu12, nvidia-cudnn-cu12). Harmless if they aren't installed."""
    if sys.platform != "win32":
        return
    for pkg in ("nvidia.cublas", "nvidia.cudnn"):
        try:
            module = importlib.import_module(pkg)
        except ImportError:
            continue
        for base in module.__path__:
            bin_dir = os.path.join(base, "bin")
            if os.path.isdir(bin_dir):
                os.add_dll_directory(bin_dir)
                os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")


def _default_compute_type(device: str) -> str:
    return "float16" if device == "cuda" else "int8"


def _warm_up(model) -> None:
    """Run one tiny transcription. A GPU that *loads* a model can still fail on
    the first real call (missing cuDNN, for example). Doing it here makes the
    failure happen at startup, where we can fall back, instead of on a user's request."""
    segments, _ = model.transcribe(np.zeros(16_000, dtype=np.float32), language="en", beam_size=1)
    list(segments)  # segments is lazy: nothing runs until you consume it


def load_model(model_size: str, device: str = "auto", compute_type: str = "auto"):
    """Return (model, device_used, compute_type_used).

    device='auto' tries CUDA, then falls back to CPU.
    device='cuda' or 'cpu' is strict: if it fails you get an error, not a silent switch.
    """
    # Imported here so that importing this module (e.g. in tests) stays fast.
    import ctranslate2
    from faster_whisper import WhisperModel

    if device not in ("auto", "cuda", "cpu"):
        raise ValueError("device must be 'auto', 'cuda' or 'cpu'")

    if device in ("auto", "cuda"):
        _register_nvidia_pip_dlls()

    candidates = ["cuda", "cpu"] if device == "auto" else [device]
    if device == "auto" and ctranslate2.get_cuda_device_count() == 0:
        logger.info("CTranslate2 sees no CUDA device; using CPU.")
        candidates = ["cpu"]

    last_error: Exception | None = None
    for dev in candidates:
        ct = _default_compute_type(dev) if compute_type == "auto" else compute_type
        try:
            logger.info("Loading model '%s' on %s (%s)...", model_size, dev, ct)
            model = WhisperModel(model_size, device=dev, compute_type=ct)
            _warm_up(model)
            logger.info("Model ready on %s (%s).", dev, ct)
            return model, dev, ct
        except Exception as exc:
            logger.warning("Could not use device '%s': %s", dev, exc)
            last_error = exc

    raise RuntimeError(f"Could not load model '{model_size}' on {candidates}.") from last_error