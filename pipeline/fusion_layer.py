"""
EchoSight AI Engine — Multimodal Fusion Layer

Computes a continuous trust score (0–1) from alignment signals:
  1. Text ↔ Rating alignment   (weight: 0.40)
  2. Image ↔ Text alignment    (weight: 0.35)
  3. Image ↔ Rating alignment  (weight: 0.25)

When no image is present, text ↔ rating gets 100% weight.
"""

import logging

logger = logging.getLogger(__name__)

# ── Weights ──────────────────────────────────────────────────────────────────
W_TEXT_RATING = 0.40
W_IMAGE_TEXT = 0.35
W_IMAGE_RATING = 0.25

# ── Thresholds ───────────────────────────────────────────────────────────────
THRESHOLD_HIGH_RISK = 0.40
THRESHOLD_MODERATE_RISK = 0.65

# ── Sentiment → expected star ────────────────────────────────────────────────
_SENTIMENT_TO_EXPECTED_RATING: dict[str, float] = {
    "positive": 4.5,
    "neutral": 3.0,
    "negative": 1.5,
}

# ── Image polarity ───────────────────────────────────────────────────────────
_IMAGE_POLARITY: dict[str, int] = {
    "damaged": -1,
    "intact": +1,
    "screenshot": 0,
    "spam": 0,
    "no_image": 0,
    "image_fetch_error": 0,
}


def _keyword_polarity(image_class: str) -> int:
    lower = image_class.lower()
    for keyword, polarity in _IMAGE_POLARITY.items():
        if keyword in lower:
            return polarity
    return 0


def _text_rating_score(sentiment: str, rating: int) -> float:
    expected = _SENTIMENT_TO_EXPECTED_RATING.get(sentiment, 3.0)
    distance = abs(rating - expected)
    return round(max(0.0, 1.0 - distance / 3.5), 4)


def _image_text_score(sentiment: str, image_class: str) -> float:
    img_pol = _keyword_polarity(image_class)
    if img_pol == 0:
        return 0.70
    sent_pol = {"positive": +1, "neutral": 0, "negative": -1}.get(sentiment, 0)
    if img_pol == sent_pol:
        return 1.0
    if sent_pol == 0:
        return 0.60
    return 0.10


def _image_rating_score(rating: int, image_class: str) -> float:
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


def evaluate_fusion(
    sentiment: str,
    image_class: str,
    rating: int,
) -> dict:
    """Compute multimodal trust score.

    Returns:
        {
            "trust_score": float (0–1),
            "risk_level": "high" | "moderate" | "low",
            "is_incongruent": bool,
            "reasoning": str,
        }
    """
    has_image = image_class not in ("no_image", "image_fetch_error", "")

    tr = _text_rating_score(sentiment, rating)

    if has_image:
        it = _image_text_score(sentiment, image_class)
        ir = _image_rating_score(rating, image_class)
        trust = round(W_TEXT_RATING * tr + W_IMAGE_TEXT * it + W_IMAGE_RATING * ir, 2)
    else:
        # No image — trust is based entirely on text ↔ rating
        it = 0.70
        ir = 0.70
        trust = round(tr, 2)

    is_incongruent = trust <= THRESHOLD_MODERATE_RISK

    # Risk level
    if trust <= THRESHOLD_HIGH_RISK:
        risk_level = "high"
    elif trust <= THRESHOLD_MODERATE_RISK:
        risk_level = "moderate"
    else:
        risk_level = "low"

    # Reasoning
    parts: list[str] = []
    if tr < 0.5:
        parts.append(
            f"Text-Rating mismatch: '{sentiment}' sentiment vs {rating}-star "
            f"(alignment: {tr:.0%})."
        )
    if has_image and it < 0.5:
        parts.append(
            f"Image-Text mismatch: '{sentiment}' text vs image='{image_class}' "
            f"(alignment: {it:.0%})."
        )
    if has_image and ir < 0.5:
        parts.append(
            f"Image-Rating mismatch: {rating}-star vs image='{image_class}' "
            f"(alignment: {ir:.0%})."
        )
    if not parts:
        parts.append("Review signals are consistent across all available modalities.")

    parts.append(f"Overall trust: {trust:.2f} — {risk_level.upper()} RISK.")
    reasoning = " ".join(parts)

    logger.info(
        "Fusion — sent=%s, img=%s, rating=%d | TR=%.2f IT=%.2f IR=%.2f → trust=%.2f (%s)",
        sentiment, image_class, rating, tr, it, ir, trust, risk_level,
    )

    return {
        "trust_score": trust,
        "risk_level": risk_level,
        "is_incongruent": is_incongruent,
        "reasoning": reasoning,
    }
