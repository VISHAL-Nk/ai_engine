"""
EchoSight Fusion Engine — TF-IDF Bot Sniper
Detects bot-farm reviews by measuring textual similarity between the
incoming review and recent reviews for the same product using TF-IDF
cosine similarity.
"""

import logging

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

# ── Thresholds ───────────────────────────────────────────────────────────────
_SIMILARITY_THRESHOLD = 0.85  # reviews above this are suspiciously similar
_MIN_REVIEWS_FOR_CHECK = 2    # need at least N existing reviews to compare


def detect_bot_farm(
    text: str,
    product_id: str,
    recent_reviews: list[str],
) -> dict:
    """Check whether *text* is a likely bot-farm duplicate.

    Compares the incoming review against *recent_reviews* (other reviews
    for the same *product_id*) using TF-IDF cosine similarity.

    Returns a dict with:
        * ``is_bot``    (bool)  – True if the review looks bot-generated
        * ``reasoning`` (str)   – Human-readable explanation
    """

    # Not enough history to make a judgement
    if len(recent_reviews) < _MIN_REVIEWS_FOR_CHECK:
        logger.debug(
            "Bot check skipped for product %s — only %d prior reviews.",
            product_id,
            len(recent_reviews),
        )
        return {
            "is_bot": False,
            "reasoning": "Insufficient review history for bot detection.",
        }

    # Build TF-IDF matrix: last entry is the incoming review
    corpus = recent_reviews + [text]
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(corpus)

    # Cosine similarity of the new review against every existing one
    incoming_vector = tfidf_matrix[-1]
    similarities = cosine_similarity(incoming_vector, tfidf_matrix[:-1]).flatten()

    max_sim = float(similarities.max())
    avg_sim = float(similarities.mean())

    logger.info(
        "Bot check for product %s — max_sim=%.4f, avg_sim=%.4f, threshold=%.2f",
        product_id,
        max_sim,
        avg_sim,
        _SIMILARITY_THRESHOLD,
    )

    if max_sim >= _SIMILARITY_THRESHOLD:
        return {
            "is_bot": True,
            "reasoning": (
                f"Bot farm detected: Review is {max_sim:.0%} similar to an "
                f"existing review for product '{product_id}'. "
                f"Threshold: {_SIMILARITY_THRESHOLD:.0%}."
            ),
        }

    return {
        "is_bot": False,
        "reasoning": "Review passed bot-farm similarity check.",
    }
