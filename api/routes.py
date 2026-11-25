"""HTTP layer only: parse the request, call the other modules, translate errors."""
import logging
import time

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile

from audio.loader import (
    AudioDecodeError, AudioError, EmptyAudioError, FFmpegNotFoundError,
    FileTooLargeError, UnsupportedFormatError, temp_audio_file,
)
from audio.preprocessor import decode_to_mono_16k, duration_seconds
from output.formatter import TranscriptionResponse, format_transcription
from recognition.language import SUPPORTED_LANGUAGES, UnsupportedLanguageError, normalize_language

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1")

_AUDIO_ERROR_STATUS = {
    UnsupportedFormatError: 415,
    EmptyAudioError: 400,
    FileTooLargeError: 413,
    AudioDecodeError: 422,
    FFmpegNotFoundError: 500,
}


@router.get("/health")
def health(request: Request):
    t = request.app.state.transcriber
    return {"status": "ok", "model": t.model_size, "device": t.device, "compute_type": t.compute_type}


@router.get("/languages")
def languages():
    return {
        "auto_detect": True,
        "languages": [{"code": c, "name": n} for c, n in SUPPORTED_LANGUAGES.items()],
    }


# A plain `def` (not `async def`) on purpose: transcription is slow and blocking.
# FastAPI runs `def` endpoints in a thread pool, so one slow request doesn't freeze the server.
@router.post("/transcribe", response_model=TranscriptionResponse)
def transcribe(
    request: Request,
    file: UploadFile = File(..., description="Audio file (wav, mp3, m4a, flac, ogg, ...)"),
    language: str | None = Query(None, description="Language code such as 'en', 'hi', 'te'. Leave empty to auto-detect."),
):
    started = time.perf_counter()
    settings = request.app.state.settings
    transcriber = request.app.state.transcriber

    try:
        language_code = normalize_language(language)
    except UnsupportedLanguageError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    try:
        with temp_audio_file(file.file, file.filename, settings.max_upload_bytes) as path:
            audio = decode_to_mono_16k(path)
        result = transcriber.transcribe(audio, language=language_code)
    except AudioError as exc:
        status = _AUDIO_ERROR_STATUS.get(type(exc), 400)
        raise HTTPException(status_code=status, detail=str(exc))
    except Exception:
        logger.exception("Unexpected error during transcription")
        raise HTTPException(status_code=500, detail="Internal error during transcription.")

    return format_transcription(
        result,
        audio_duration=duration_seconds(audio),
        processing_time=time.perf_counter() - started,
        language_source="explicit" if language_code else "auto",
        model=transcriber.model_size,
        device=transcriber.device,
    )