"""
EchoSight AI Engine — Text Sentiment Analysis (NLP)
Uses a fine-tuned RoBERTa model to classify review text as
positive, neutral, or negative. Returns confidence scores.
"""

import logging

from transformers import pipeline  # type: ignore

logger = logging.getLogger(__name__)

# ── Model initialisation (lazy — loaded on first call) ───────────────────────
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

_sentiment_pipeline = None


def _get_pipeline():
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
        logger.info("Loading NLP sentiment model: %s …", MODEL_NAME)
        _sentiment_pipeline = pipeline(
            task="sentiment-analysis",
            model=MODEL_NAME,
            tokenizer=MODEL_NAME,
            top_k=None,
        )
        logger.info("NLP model loaded successfully.")
    return _sentiment_pipeline


# ── Label mapping ────────────────────────────────────────────────────────────
_LABEL_MAP: dict[str, str] = {
    "positive": "positive",
    "neutral": "neutral",
    "negative": "negative",
    "pos": "positive",
    "neu": "neutral",
    "neg": "negative",
}


def analyze_text(text: str) -> dict:
    """Return sentiment analysis with confidence scores.

    Returns:
        {
            "sentiment": "positive" | "neutral" | "negative",
            "confidence": float (0–1),
            "all_scores": {"positive": float, "neutral": float, "negative": float}
        }
    """
    if not text or not text.strip():
        return {
            "sentiment": "neutral",
            "confidence": 0.5,
            "all_scores": {"positive": 0.0, "neutral": 1.0, "negative": 0.0},
        }

    pipe = _get_pipeline()
    results = pipe(text, truncation=True, max_length=512)

    # results is [[{label, score}, …]] when top_k is set
    scores = results[0] if isinstance(results[0], list) else results

    # Build scores dict and find the best
    all_scores: dict[str, float] = {}
    best_label = "neutral"
    best_score = 0.0

    for item in scores:
        raw_label = item["label"].lower().strip()
        mapped = _LABEL_MAP.get(raw_label, raw_label)
        all_scores[mapped] = round(item["score"], 4)
        if item["score"] > best_score:
            best_score = item["score"]
            best_label = mapped

    logger.debug(
        "Text sentiment: %s (confidence=%.4f, all=%s)",
        best_label,
        best_score,
        all_scores,
    )

    return {
        "sentiment": best_label,
        "confidence": round(best_score, 4),
        "all_scores": all_scores,
    }
