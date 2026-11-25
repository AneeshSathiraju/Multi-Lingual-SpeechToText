"""The shape of the API's JSON response."""
from typing import Literal

from pydantic import BaseModel

from recognition.language import language_name
from recognition.transcriber import TranscriptionResult


class SegmentOut(BaseModel):
    start: float
    end: float
    text: str


class TranscriptionResponse(BaseModel):
    transcript: str
    language: str
    language_name: str
    language_source: Literal["auto", "explicit"]
    language_probability: float | None
    audio_duration_seconds: float
    processing_time_seconds: float
    model: str
    device: str
    segments: list[SegmentOut]


def format_transcription(
    result: TranscriptionResult,
    *,
    audio_duration: float,
    processing_time: float,
    language_source: Literal["auto", "explicit"],
    model: str,
    device: str,
) -> TranscriptionResponse:
    return TranscriptionResponse(
        transcript=result.text,
        language=result.language,
        language_name=language_name(result.language),
        language_source=language_source,
        language_probability=(
            round(result.language_probability, 3) if result.language_probability is not None else None
        ),
        audio_duration_seconds=round(audio_duration, 2),
        processing_time_seconds=round(processing_time, 2),
        model=model,
        device=device,
        segments=[SegmentOut(start=round(s.start, 2), end=round(s.end, 2), text=s.text) for s in result.segments],
    )