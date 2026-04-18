"""
EchoSight AI Engine — Review Bomb Detector

Detects coordinated review-bombing by analysing temporal patterns:
  1. Volume spike — too many reviews in a short time window
  2. Rating cluster — most reviews converge on the same rating
  3. Sentiment cluster — most reviews share the same sentiment
"""

import logging
from datetime import datetime, timedelta, timezone
from collections import Counter

logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────────────
TIME_WINDOW_HOURS = 24
MIN_REVIEWS_FOR_FLAG = 5
RATING_CLUSTER_THRESHOLD = 0.70
SENTIMENT_CLUSTER_THRESHOLD = 0.70


def _to_utc_aware(value: datetime | None) -> datetime | None:
    """Normalize datetimes so comparisons are always timezone-safe."""
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def detect_review_bomb(
    current_timestamp: datetime,
    current_rating: int,
    current_sentiment: str,
    recent_reviews: list[dict],
) -> dict:
    """Detect review-bombing from temporal + pattern signals.

    *recent_reviews* is a list of dicts with:
        * ``timestamp`` (datetime)
        * ``rating``    (int)
        * ``sentiment`` (str)

    Returns:
        {
            "is_review_bomb": bool,
            "bomb_type": str,
            "reasoning": str,
        }
    """
    current_ts = _to_utc_aware(current_timestamp) or datetime.now(timezone.utc)
    cutoff = current_ts - timedelta(hours=TIME_WINDOW_HOURS)

    windowed = []
    for review in recent_reviews:
        ts = _to_utc_aware(review.get("timestamp"))
        if not ts or ts < cutoff:
            continue

        windowed.append({
            "timestamp": ts,
            "rating": review.get("rating", 3),
            "sentiment": review.get("sentiment", "neutral"),
        })

    windowed.append({
        "timestamp": current_ts,
        "rating": current_rating,
        "sentiment": current_sentiment,
    })

    total = len(windowed)

    if total < MIN_REVIEWS_FOR_FLAG:
        return {
            "is_review_bomb": False,
            "bomb_type": "none",
            "reasoning": (
                f"Only {total} review(s) in the last {TIME_WINDOW_HOURS}h — "
                f"below the {MIN_REVIEWS_FOR_FLAG}-review threshold."
            ),
        }

    # Rating distribution
    ratings = [r["rating"] for r in windowed]
    rating_counts = Counter(ratings)
    most_common_rating, most_common_count = rating_counts.most_common(1)[0]
    rating_ratio = most_common_count / total

    # Sentiment distribution
    sentiments = [r["sentiment"] for r in windowed]
    sentiment_counts = Counter(sentiments)
    most_common_sentiment, sent_common_count = sentiment_counts.most_common(1)[0]
    sentiment_ratio = sent_common_count / total

    logger.info(
        "Bomb check — %d reviews in %dh | Rating: %d★ at %.0f%% | Sentiment: '%s' at %.0f%%",
        total, TIME_WINDOW_HOURS,
        most_common_rating, rating_ratio * 100,
        most_common_sentiment, sentiment_ratio * 100,
    )

    is_bomb = (
        rating_ratio >= RATING_CLUSTER_THRESHOLD
        and sentiment_ratio >= SENTIMENT_CLUSTER_THRESHOLD
    )

    if not is_bomb:
        return {
            "is_review_bomb": False,
            "bomb_type": "none",
            "reasoning": (
                f"{total} reviews in {TIME_WINDOW_HOURS}h but no strong clustering "
                f"(rating: {rating_ratio:.0%}, sentiment: {sentiment_ratio:.0%})."
            ),
        }

    if most_common_rating <= 2 and most_common_sentiment == "negative":
        bomb_type = "negative_bomb"
        reasoning = (
            f"⚠ NEGATIVE review bomb: {total} reviews in {TIME_WINDOW_HOURS}h — "
            f"{rating_ratio:.0%} are {most_common_rating}★ and "
            f"{sentiment_ratio:.0%} are negative."
        )
    elif most_common_rating >= 4 and most_common_sentiment == "positive":
        bomb_type = "positive_bomb"
        reasoning = (
            f"⚠ POSITIVE review bomb: {total} reviews in {TIME_WINDOW_HOURS}h — "
            f"{rating_ratio:.0%} are {most_common_rating}★ and "
            f"{sentiment_ratio:.0%} are positive."
        )
    else:
        bomb_type = "suspicious_cluster"
        reasoning = (
            f"⚠ Suspicious cluster: {total} reviews in {TIME_WINDOW_HOURS}h with "
            f"{rating_ratio:.0%} rating convergence on {most_common_rating}★."
        )

    return {
        "is_review_bomb": True,
        "bomb_type": bomb_type,
        "reasoning": reasoning,
    }
