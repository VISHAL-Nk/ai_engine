"""
EchoSight Fusion Engine — Review Bomb Detector

Detects coordinated review-bombing by analyzing temporal patterns
in recent reviews for the same product:

  1. Volume spike  — Too many reviews arriving in a short time window
  2. Rating cluster — Most reviews converge on the same rating
  3. Sentiment cluster — Most reviews share the same sentiment

A review bomb can be NEGATIVE (mass 1-star attacks to tank a product)
or POSITIVE (mass 5-star fake praise to boost a product).
"""

import logging
from datetime import datetime, timedelta
from collections import Counter

logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────────────
TIME_WINDOW_HOURS = 24          # analyse reviews within this window
MIN_REVIEWS_FOR_FLAG = 5        # need at least this many reviews to trigger
RATING_CLUSTER_THRESHOLD = 0.70  # 70%+ same rating = suspicious
SENTIMENT_CLUSTER_THRESHOLD = 0.70  # 70%+ same sentiment = suspicious


def detect_review_bomb(
    current_timestamp: datetime,
    current_rating: int,
    current_sentiment: str,
    recent_reviews: list[dict],
) -> dict:
    """Detect review-bombing from temporal + pattern signals.

    *recent_reviews* is a list of dicts, each containing:
        * ``timestamp`` (datetime) — when the review was posted
        * ``rating``    (int)      — star rating 1–5
        * ``sentiment`` (str)      — "positive", "neutral", or "negative"

    Returns a dict with:
        * ``is_review_bomb`` (bool)
        * ``bomb_type``      (str)  — "negative_bomb", "positive_bomb", or "none"
        * ``reasoning``      (str)
    """

    # ── Include the current review in the analysis window ────────────────
    cutoff = current_timestamp - timedelta(hours=TIME_WINDOW_HOURS)

    windowed = [
        r for r in recent_reviews
        if r.get("timestamp") and r["timestamp"] >= cutoff
    ]

    # Add the incoming review itself
    windowed.append({
        "timestamp": current_timestamp,
        "rating": current_rating,
        "sentiment": current_sentiment,
    })

    total = len(windowed)

    # ── Not enough volume to be a bomb ───────────────────────────────────
    if total < MIN_REVIEWS_FOR_FLAG:
        return {
            "is_review_bomb": False,
            "bomb_type": "none",
            "reasoning": (
                f"Only {total} review(s) in the last {TIME_WINDOW_HOURS}h — "
                f"below the {MIN_REVIEWS_FOR_FLAG}-review threshold for "
                f"bomb detection."
            ),
        }

    # ── Analyse rating distribution ──────────────────────────────────────
    ratings = [r["rating"] for r in windowed]
    rating_counts = Counter(ratings)
    most_common_rating, most_common_count = rating_counts.most_common(1)[0]
    rating_ratio = most_common_count / total

    # ── Analyse sentiment distribution ───────────────────────────────────
    sentiments = [r["sentiment"] for r in windowed]
    sentiment_counts = Counter(sentiments)
    most_common_sentiment, sent_common_count = sentiment_counts.most_common(1)[0]
    sentiment_ratio = sent_common_count / total

    logger.info(
        "Bomb check — %d reviews in %dh window | "
        "Rating cluster: %d★ at %.0f%% | Sentiment cluster: '%s' at %.0f%%",
        total, TIME_WINDOW_HOURS,
        most_common_rating, rating_ratio * 100,
        most_common_sentiment, sentiment_ratio * 100,
    )

    # ── Decision: both rating AND sentiment must cluster ─────────────────
    is_bomb = (
        rating_ratio >= RATING_CLUSTER_THRESHOLD
        and sentiment_ratio >= SENTIMENT_CLUSTER_THRESHOLD
    )

    if not is_bomb:
        return {
            "is_review_bomb": False,
            "bomb_type": "none",
            "reasoning": (
                f"{total} reviews in {TIME_WINDOW_HOURS}h but no strong "
                f"clustering detected (rating cluster: {rating_ratio:.0%}, "
                f"sentiment cluster: {sentiment_ratio:.0%})."
            ),
        }

    # ── Determine bomb direction ─────────────────────────────────────────
    if most_common_rating <= 2 and most_common_sentiment == "negative":
        bomb_type = "negative_bomb"
        reasoning = (
            f"⚠ NEGATIVE review bomb detected: {total} reviews in the last "
            f"{TIME_WINDOW_HOURS}h — {rating_ratio:.0%} are {most_common_rating}★ "
            f"and {sentiment_ratio:.0%} are negative. Coordinated attack "
            f"likely aimed at demoting this product."
        )
    elif most_common_rating >= 4 and most_common_sentiment == "positive":
        bomb_type = "positive_bomb"
        reasoning = (
            f"⚠ POSITIVE review bomb detected: {total} reviews in the last "
            f"{TIME_WINDOW_HOURS}h — {rating_ratio:.0%} are {most_common_rating}★ "
            f"and {sentiment_ratio:.0%} are positive. Coordinated fake praise "
            f"likely aimed at boosting this product."
        )
    else:
        bomb_type = "suspicious_cluster"
        reasoning = (
            f"⚠ Suspicious review cluster: {total} reviews in {TIME_WINDOW_HOURS}h "
            f"with {rating_ratio:.0%} rating convergence on {most_common_rating}★ "
            f"and {sentiment_ratio:.0%} sentiment convergence on '{most_common_sentiment}'."
        )

    return {
        "is_review_bomb": True,
        "bomb_type": bomb_type,
        "reasoning": reasoning,
    }
