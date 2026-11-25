"""Validate an uploaded file and store it safely on disk. Knows nothing about FastAPI or Whisper."""
import contextlib
import os
import tempfile
from pathlib import Path
from typing import BinaryIO, Iterator

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".webm", ".aac", ".wma"}
_CHUNK_SIZE = 1024 * 1024  # read 1 MB at a time so big uploads never sit fully in memory


class AudioError(Exception):
    """Base class. The API layer maps each subclass to an HTTP status."""


class UnsupportedFormatError(AudioError): ...
class EmptyAudioError(AudioError): ...
class FileTooLargeError(AudioError): ...
class AudioDecodeError(AudioError): ...
class FFmpegNotFoundError(AudioError): ...


@contextlib.contextmanager
def temp_audio_file(fileobj: BinaryIO, filename: str | None, max_bytes: int) -> Iterator[Path]:
    """Save an upload to a temp file, yield its path, and always delete it afterwards.

    We keep only the *extension* of the client's filename. The name itself is never
    used for a path, so a malicious name like '..\\..\\x.wav' can't do anything.
    The extension check is only a quick first filter; the real test is whether
    FFmpeg can decode the file (see preprocessor.py).
    """
    suffix = Path(filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise UnsupportedFormatError(
            f"Unsupported file type '{suffix or 'none'}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )

    fd, name = tempfile.mkstemp(suffix=suffix)
    path = Path(name)
    try:
        written = 0
        with os.fdopen(fd, "wb") as out:  # closed before we yield (Windows needs this)
            while chunk := fileobj.read(_CHUNK_SIZE):
                written += len(chunk)
                if written > max_bytes:
                    raise FileTooLargeError(f"File exceeds the {max_bytes // (1024 * 1024)} MB limit.")
                out.write(chunk)
        if written == 0:
            raise EmptyAudioError("The uploaded file is empty.")
        yield path
    finally:
        path.unlink(missing_ok=True)