"""
EchoSight AI Engine — Text Preprocessor

Cleans raw review text:
  1. Language detection  (langdetect)
  2. Translation         (deep-translator → Google Translate)
  3. Emoji normalisation (emoji → text descriptors)
  4. Spell correction    (SymSpell)
  5. Noise stripping     (unicode, punctuation, whitespace)
"""

import logging
import re

logger = logging.getLogger(__name__)

# ── Lazy-loaded deps ─────────────────────────────────────────────────────────
_translator = None
_sym_spell = None


def _get_translator():
    global _translator
    if _translator is None:
        try:
            from deep_translator import GoogleTranslator
            _translator = GoogleTranslator(source="auto", target="en")
            logger.info("Google Translator initialised.")
        except ImportError:
            logger.warning("deep-translator not installed — translation disabled.")
    return _translator


def _get_sym_spell():
    global _sym_spell
    if _sym_spell is None:
        try:
            from symspellpy import SymSpell
            from pathlib import Path
            import symspellpy as _pkg

            _sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
            dict_file = Path(_pkg.__file__).parent / "frequency_dictionary_en_82_765.txt"
            if dict_file.exists():
                _sym_spell.load_dictionary(str(dict_file), term_index=0, count_index=1)
                logger.info("SymSpell dictionary loaded.")
            else:
                logger.warning("SymSpell dictionary not found — spell check disabled.")
                _sym_spell = None
        except ImportError:
            logger.warning("symspellpy not installed — spell check disabled.")
    return _sym_spell


# ── Public functions ─────────────────────────────────────────────────────────


def detect_language(text: str) -> str:
    """Detect language. Returns ISO 639-1 code."""
    try:
        from langdetect import detect as _detect
        return _detect(text)
    except Exception:
        return "en"


def translate_to_english(text: str, source_lang: str = "auto") -> str:
    """Translate to English. Returns original on failure."""
    if source_lang == "en":
        return text
    translator = _get_translator()
    if translator is None:
        return text
    try:
        result = translator.translate(text)
        return result if result else text
    except Exception:
        logger.warning("Translation failed for: %.60s…", text)
        return text


def normalize_emojis(text: str) -> str:
    """Convert emojis to text descriptors."""
    try:
        import emoji
        return emoji.demojize(text, delimiters=(" ", " "))
    except ImportError:
        return text


def fix_typos(text: str) -> str:
    """Spell correction using SymSpell."""
    spell = _get_sym_spell()
    if spell is None:
        return text
    try:
        suggestions = spell.lookup_compound(text, max_edit_distance=2)
        if suggestions:
            return suggestions[0].term
    except Exception:
        pass
    return text


def strip_noise(text: str) -> str:
    """Remove unicode artefacts, collapse punctuation, normalise whitespace."""
    text = text.encode("ascii", errors="ignore").decode("ascii")
    text = re.sub(r"([!?.])\1{2,}", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_review(raw_text: str) -> dict:
    """Full preprocessing pipeline.

    Returns:
        {
            "detected_language": str,
            "cleaned_text": str,
            "preprocessing_flags": [str, ...],
        }
    """
    flags: list[str] = []

    if not raw_text or not raw_text.strip():
        return {
            "detected_language": "en",
            "cleaned_text": "",
            "preprocessing_flags": ["empty_review"],
        }

    # 1. Language detection
    detected_lang = detect_language(raw_text)
    if detected_lang != "en":
        flags.append("mixed_language")

    # 2. Emojis (before translation)
    text = normalize_emojis(raw_text)

    # 3. Translate if needed
    if detected_lang != "en":
        text = translate_to_english(text, source_lang=detected_lang)
        flags.append(f"translated_from_{detected_lang}")

    # 4. Noise stripping
    text = strip_noise(text)

    # 5. Spell correction
    text = fix_typos(text)

    # 6. Quality flags
    word_count = len(text.split())
    if word_count < 3:
        flags.append("very_short")
    if word_count > 500:
        flags.append("very_long")

    return {
        "detected_language": detected_lang,
        "cleaned_text": text,
        "preprocessing_flags": flags,
    }
