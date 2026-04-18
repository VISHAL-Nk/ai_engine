"""
EchoSight Fusion Engine — Text Sentiment Analysis (NLP)
Uses a fine-tuned RoBERTa model to classify review text as
positive, neutral, or negative.
"""

import logging

from transformers import pipeline  # type: ignore

logger = logging.getLogger(__name__)

# ── Model initialisation (runs once at import / first worker start) ──────────
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

logger.info("Loading NLP sentiment model: %s …", MODEL_NAME)
_sentiment_pipeline = pipeline(
    task="sentiment-analysis",
    model=MODEL_NAME,
    tokenizer=MODEL_NAME,
    top_k=None,  # return scores for all labels
)
logger.info("NLP model loaded successfully.")


# ── Label mapping ────────────────────────────────────────────────────────────
# The RoBERTa model emits labels like "positive", "neutral", "negative" (or
# sometimes uppercased).  We normalise to lowercase for downstream consumers.
_LABEL_MAP: dict[str, str] = {
    "positive": "positive",
    "neutral": "neutral",
    "negative": "negative",
    # Some model revisions use abbreviated labels:
    "pos": "positive",
    "neu": "neutral",
    "neg": "negative",
}


def analyze_text(text: str) -> str:
    """Return the dominant sentiment label for *text*.

    Returns one of ``"positive"``, ``"neutral"``, or ``"negative"``.
    """
    if not text or not text.strip():
        return "neutral"

    results = _sentiment_pipeline(text, truncation=True, max_length=512)

    # `results` is a list of lists when top_k is set:  [[{label, score}, …]]
    scores = results[0] if isinstance(results[0], list) else results

    best = max(scores, key=lambda x: x["score"])
    raw_label: str = best["label"].lower().strip()

    sentiment = _LABEL_MAP.get(raw_label, raw_label)
    logger.debug("Text sentiment: %s (raw=%s, score=%.4f)", sentiment, raw_label, best["score"])
    return sentiment
