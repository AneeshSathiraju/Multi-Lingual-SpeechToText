"""Languages this API accepts for explicit selection."""

# A subset of Whisper's ~99 languages, focused on India plus English.
# "Supported by Whisper" does not mean equal accuracy across languages.
SUPPORTED_LANGUAGES: dict[str, str] = {
    "en": "English",
    "hi": "Hindi",
    "te": "Telugu",
    "ta": "Tamil",
    "kn": "Kannada",
    "ml": "Malayalam",
    "mr": "Marathi",
    "bn": "Bengali",
    "gu": "Gujarati",
    "pa": "Punjabi",
    "ur": "Urdu",
}


class UnsupportedLanguageError(ValueError):
    pass


def normalize_language(code: str | None) -> str | None:
    """Return a valid language code, or None meaning 'detect automatically'."""
    if code is None:
        return None
    code = code.strip().lower()
    if code in ("", "auto"):  # Swagger UI sometimes sends an empty string
        return None
    if code not in SUPPORTED_LANGUAGES:
        raise UnsupportedLanguageError(
            f"Unsupported language '{code}'. Supported: {', '.join(SUPPORTED_LANGUAGES)} (or omit for auto-detect)."
        )
    return code


def language_name(code: str) -> str:
    # Auto-detection can return any Whisper language, not only the ones above.
    return SUPPORTED_LANGUAGES.get(code, code)