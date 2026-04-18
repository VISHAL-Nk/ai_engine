"""
EchoSight Fusion Engine — Multimodal Fusion Layer (v3)

Computes a continuous trust score (0–1) from three independent
alignment signals:

  1. Text ↔ Rating alignment   (weight: 0.40)
  2. Image ↔ Text alignment    (weight: 0.35)
  3. Image ↔ Rating alignment  (weight: 0.25)

Review-bomb detection is handled separately by review_bomb_detector.py.
"""

import logging

logger = logging.getLogger(__name__)

# ── Weights for the three alignment signals ──────────────────────────────────
W_TEXT_RATING = 0.40
W_IMAGE_TEXT = 0.35
W_IMAGE_RATING = 0.25

# ── Risk thresholds ──────────────────────────────────────────────────────────
THRESHOLD_HIGH_RISK = 0.40
THRESHOLD_MODERATE_RISK = 0.65

# ── Sentiment ↔ expected star mapping ────────────────────────────────────────
_SENTIMENT_TO_EXPECTED_RATING: dict[str, float] = {
    "positive": 4.5,
    "neutral": 3.0,
    "negative": 1.5,
}

# ── Image class keywords ↔ expected polarity ─────────────────────────────────
_IMAGE_POLARITY: dict[str, int] = {
    "damaged": -1,
    "intact": +1,
    "screenshot": 0,
    "spam": 0,
    "no_image": 0,
    "image_fetch_error": 0,
}


def _keyword_polarity(image_class: str) -> int:
    """Extract polarity from the image classification label."""
    lower = image_class.lower()
    for keyword, polarity in _IMAGE_POLARITY.items():
        if keyword in lower:
            return polarity
    return 0


def _text_rating_score(sentiment: str, rating: int) -> float:
    """How well the text sentiment aligns with the star rating (0–1)."""
    expected = _SENTIMENT_TO_EXPECTED_RATING.get(sentiment, 3.0)
    distance = abs(rating - expected)
    return round(max(0.0, 1.0 - distance / 3.5), 4)


def _image_text_score(sentiment: str, image_class: str) -> float:
    """How well the image evidence aligns with the text sentiment (0–1)."""
    img_pol = _keyword_polarity(image_class)

    if img_pol == 0:
        return 0.70  # ambiguous / no image

    sent_pol = {"positive": +1, "neutral": 0, "negative": -1}.get(sentiment, 0)

    if img_pol == sent_pol:
        return 1.0
    if sent_pol == 0:
        return 0.60
    return 0.10  # direct contradiction


def _image_rating_score(rating: int, image_class: str) -> float:
    """How well the image evidence aligns with the star rating (0–1)."""
    img_pol = _keyword_polarity(image_class)

    if img_pol == 0:
        return 0.70

    if rating >= 4:
        rating_pol = +1
    elif rating <= 2:
        rating_pol = -1
    else:
        rating_pol = 0

    if img_pol == rating_pol:
        return 1.0
    if rating_pol == 0:
        return 0.60
    return 0.10


def _build_reasoning(
    tr_score: float,
    it_score: float,
    ir_score: float,
    trust: float,
    sentiment: str,
    image_class: str,
    rating: int,
) -> str:
    """Build a human-readable reasoning string."""
    parts: list[str] = []

    if tr_score < 0.5:
        parts.append(
            f"Text-Rating mismatch: '{sentiment}' sentiment vs {rating}-star "
            f"(alignment: {tr_score:.0%})."
        )
    if it_score < 0.5:
        parts.append(
            f"Image-Text mismatch: '{sentiment}' text vs image='{image_class}' "
            f"(alignment: {it_score:.0%})."
        )
    if ir_score < 0.5:
        parts.append(
            f"Image-Rating mismatch: {rating}-star vs image='{image_class}' "
            f"(alignment: {ir_score:.0%})."
        )

    if not parts:
        parts.append("Review signals are consistent across text, image, and rating.")

    if trust <= THRESHOLD_HIGH_RISK:
        parts.append(f"Overall trust: {trust:.2f} — HIGH RISK.")
    elif trust <= THRESHOLD_MODERATE_RISK:
        parts.append(f"Overall trust: {trust:.2f} — MODERATE RISK.")
    else:
        parts.append(f"Overall trust: {trust:.2f} — LOW RISK.")

    return " ".join(parts)


def evaluate_fusion(sentiment: str, image_class: str, rating: int) -> dict:
    """Evaluate cross-modal congruence using a weighted scoring system.

    Returns a dict with keys:
        * ``is_incongruent`` (bool)
        * ``trust_score``    (float, 0–1)
        * ``reasoning``      (str)
    """

    tr = _text_rating_score(sentiment, rating)
    it = _image_text_score(sentiment, image_class)
    ir = _image_rating_score(rating, image_class)

    trust = round(W_TEXT_RATING * tr + W_IMAGE_TEXT * it + W_IMAGE_RATING * ir, 2)
    is_incongruent = trust <= THRESHOLD_MODERATE_RISK

    reasoning = _build_reasoning(tr, it, ir, trust, sentiment, image_class, rating)

    result = {
        "is_incongruent": is_incongruent,
        "trust_score": trust,
        "reasoning": reasoning,
    }

    logger.info(
        "Fusion — sent=%s, img=%s, rating=%d | "
        "TR=%.2f IT=%.2f IR=%.2f → trust=%.2f",
        sentiment, image_class, rating, tr, it, ir, trust,
    )
    return result
