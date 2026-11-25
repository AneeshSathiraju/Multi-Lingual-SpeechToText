import pytest

from recognition.language import SUPPORTED_LANGUAGES, UnsupportedLanguageError, normalize_language


@pytest.mark.parametrize("value", [None, "", "auto", "AUTO", "  auto "])
def test_auto_values_mean_autodetect(value):
    assert normalize_language(value) is None


def test_codes_are_case_insensitive():
    assert normalize_language("HI") == "hi"


def test_unknown_language_rejected():
    with pytest.raises(UnsupportedLanguageError):
        normalize_language("xx")


def test_target_languages_present():
    assert {"en", "hi", "te"} <= set(SUPPORTED_LANGUAGES)