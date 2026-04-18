"""
EchoSight AI Engine — Bot Sniper (TF-IDF + Cross-Product Account Check)

Two checks:
  A. Single-product: TF-IDF cosine similarity against recent reviews for same product
  B. Cross-product: Account-level pattern detection across all products
"""

import logging
from collections import Counter
import re
from datetime import datetime, timezone

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

# ── Thresholds ───────────────────────────────────────────────────────────────
_SIMILARITY_THRESHOLD = 0.35
_MIN_REVIEWS_FOR_CHECK = 1
_SPAM_VELOCITY_LIMIT = 10  # max reviews per 24 hours per account

_PROMO_STRONG_PATTERNS: list[tuple[str, str]] = [
    ("cta_link", r"\bclick\s+(this\s+)?link\b"),
    ("buy_now", r"\bbuy\s+now\b"),
    ("cash_reward", r"\bcash\s+reward\b"),
    ("instant_payout", r"\b(instant\s+cash|instant\s+payout|direct\s+payout)\b"),
    ("referral", r"\breferral\s+code\b"),
    ("private_channel", r"\bprivate\s+channel\b"),
    ("guaranteed_profit", r"\b(guaranteed\s+profit|guaranteed\s+returns?)\b"),
]

_PROMO_WEAK_PATTERNS: list[tuple[str, str]] = [
    ("limited_offer", r"\blimited[-\s]*time\s+offer\b"),
    ("offer_now", r"\boffer\s+now\b"),
    ("visit_now", r"\bvisit\s+now\b"),
    ("unreal_claim", r"\b\d{3,}%\b"),
]

_URL_PATTERN = re.compile(r"https?://|www\.", re.IGNORECASE)


def _normalize_for_exact_match(value: str) -> str:
    """Normalize text for strict duplicate matching."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", value.lower())).strip()


def _to_utc_aware(value: datetime | None) -> datetime | None:
    """Normalize datetimes so arithmetic is timezone-safe."""
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _detect_promotional_spam(text: str) -> tuple[bool, str]:
    """Detect obvious promotional/ad-like review spam patterns."""
    normalized = _normalize_for_exact_match(text)
    if not normalized:
        return False, ""

    strong_hits = [
        name for name, pattern in _PROMO_STRONG_PATTERNS
        if re.search(pattern, normalized, flags=re.IGNORECASE)
    ]
    weak_hits = [
        name for name, pattern in _PROMO_WEAK_PATTERNS
        if re.search(pattern, normalized, flags=re.IGNORECASE)
    ]
    has_url = bool(_URL_PATTERN.search(text))

    should_flag = (
        (len(strong_hits) >= 2)
        or (len(strong_hits) >= 1 and len(weak_hits) >= 1)
        or (len(strong_hits) >= 1 and has_url)
        or (has_url and len(weak_hits) >= 2)
    )

    if not should_flag:
        return False, ""

    hit_summary = ", ".join(strong_hits + weak_hits) if (strong_hits or weak_hits) else "url_pattern"
    reasoning = (
        "Promotional spam detected: ad-like call-to-action language identified "
        f"(signals: {hit_summary})."
    )
    return True, reasoning


def detect_bot_farm(
    text: str,
    product_id: str,
    recent_reviews: list[dict],
) -> dict:
    """Check whether *text* is a bot-farm duplicate for the same product.

    *recent_reviews* is a list of dicts, each with:
        * ``_id``   (str) — review ObjectId
        * ``text``  (str) — review text

    Returns:
        {
            "is_bot": bool,
            "is_duplicate": bool,
            "duplicate_of": str | None,      ← ObjectId of matched review
            "similar_reviews": [str, ...],    ← ObjectIds of all similar reviews
            "max_similarity": float,
            "reasoning": str,
        }
    """
    is_promotional, promo_reason = _detect_promotional_spam(text)
    if is_promotional:
        return {
            "is_bot": True,
            "is_duplicate": False,
            "duplicate_of": None,
            "similar_reviews": [],
            "max_similarity": 0.0,
            "reasoning": promo_reason,
        }

    if len(recent_reviews) < _MIN_REVIEWS_FOR_CHECK:
        return {
            "is_bot": False,
            "is_duplicate": False,
            "duplicate_of": None,
            "similar_reviews": [],
            "max_similarity": 0.0,
            "reasoning": "Insufficient review history for bot detection.",
        }

    # Fast-path exact duplicate check after normalization.
    normalized_incoming = _normalize_for_exact_match(text)
    if normalized_incoming:
        exact_match_ids = [
            str(r["_id"])
            for r in recent_reviews
            if _normalize_for_exact_match(r.get("text", "")) == normalized_incoming
        ]
        if exact_match_ids:
            duplicate_of = exact_match_ids[0]
            return {
                "is_bot": True,
                "is_duplicate": True,
                "duplicate_of": duplicate_of,
                "similar_reviews": exact_match_ids,
                "max_similarity": 1.0,
                "reasoning": (
                    "Bot farm detected: Exact normalized text match found with "
                    f"{len(exact_match_ids)} existing review(s)."
                ),
            }

    # Build TF-IDF matrix
    corpus_texts = [r["text"] for r in recent_reviews] + [text]
    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(corpus_texts)
    except ValueError:
        # Fallback for tiny/stopword-only texts where TF-IDF has empty vocabulary.
        return {
            "is_bot": False,
            "is_duplicate": False,
            "duplicate_of": None,
            "similar_reviews": [],
            "max_similarity": 0.0,
            "reasoning": "Insufficient lexical signal for TF-IDF duplicate check.",
        }

    # Cosine similarity of new review against all existing
    incoming_vector = tfidf_matrix[-1]
    similarities = cosine_similarity(incoming_vector, tfidf_matrix[:-1]).flatten()

    max_sim = float(similarities.max())
    max_idx = int(similarities.argmax())

    # Find all reviews above threshold
    similar_ids = []
    for idx, sim in enumerate(similarities):
        if sim >= _SIMILARITY_THRESHOLD:
            similar_ids.append(str(recent_reviews[idx]["_id"]))

    logger.info(
        "Bot check for product %s — max_sim=%.4f, threshold=%.2f, matches=%d",
        product_id, max_sim, _SIMILARITY_THRESHOLD, len(similar_ids),
    )

    if max_sim >= _SIMILARITY_THRESHOLD:
        duplicate_of = str(recent_reviews[max_idx]["_id"])
        return {
            "is_bot": True,
            "is_duplicate": True,
            "duplicate_of": duplicate_of,
            "similar_reviews": similar_ids,
            "max_similarity": round(max_sim, 4),
            "reasoning": (
                f"Bot farm detected: Review is {max_sim:.0%} similar to review "
                f"'{duplicate_of}'. Threshold: {_SIMILARITY_THRESHOLD:.0%}."
            ),
        }

    return {
        "is_bot": False,
        "is_duplicate": False,
        "duplicate_of": None,
        "similar_reviews": [],
        "max_similarity": round(max_sim, 4),
        "reasoning": "Review passed bot-farm similarity check.",
    }


def check_account_spam(
    customer_id: str,
    account_reviews: list[dict],
) -> dict:
    """Cross-product account-level spam detection.

    *account_reviews* is a list of dicts, each with:
        * ``product_id``  (str)
        * ``rating``      (int)
        * ``text``        (str)
        * ``created_at``  (datetime)
        * ``category``    (str)

    Returns:
        {
            "account_flags": [str, ...],   ← e.g. ["spam_velocity", "account_manipulation"]
            "reasoning": str,
        }
    """
    flags: list[str] = []
    reasons: list[str] = []

    if len(account_reviews) < 2:
        return {"account_flags": [], "reasoning": "Insufficient account history."}

    # ── Check 1: Velocity — too many reviews in 24 hours ─────────────────
    now = datetime.now(timezone.utc)
    recent_24h = []
    for review in account_reviews:
        created_at = _to_utc_aware(review.get("created_at"))
        if not created_at:
            continue
        if (now - created_at).total_seconds() < 86400:
            recent_24h.append(review)

    if len(recent_24h) > _SPAM_VELOCITY_LIMIT:
        flags.append("spam_velocity")
        reasons.append(
            f"Account posted {len(recent_24h)} reviews in 24 hours "
            f"(limit: {_SPAM_VELOCITY_LIMIT})."
        )

    # ── Check 2: Cross-product text similarity ───────────────────────────
    if len(account_reviews) >= 3:
        texts = [r["text"] for r in account_reviews]
        try:
            vectorizer = TfidfVectorizer(stop_words="english")
            tfidf = vectorizer.fit_transform(texts)
            sim_matrix = cosine_similarity(tfidf)

            # Average pairwise similarity (exclude diagonal)
            n = len(texts)
            total_sim = 0.0
            count = 0
            for i in range(n):
                for j in range(i + 1, n):
                    total_sim += sim_matrix[i][j]
                    count += 1
            avg_sim = total_sim / count if count > 0 else 0.0

            if avg_sim > 0.70:
                flags.append("cross_product_duplicate")
                reasons.append(
                    f"Average text similarity across account reviews: {avg_sim:.0%} "
                    f"(suspicious copy-paste pattern)."
                )
        except Exception:
            pass  # TF-IDF can fail on very short texts

    # ── Check 3: Rating manipulation ─────────────────────────────────────
    # Group reviews by category, check if boosting one product and tanking others
    category_groups: dict[str, list[dict]] = {}
    for r in account_reviews:
        cat = r.get("category", "unknown")
        category_groups.setdefault(cat, []).append(r)

    for category, reviews in category_groups.items():
        if len(reviews) < 3:
            continue

        product_ratings: dict[str, list[int]] = {}
        for r in reviews:
            pid = str(r["product_id"])
            product_ratings.setdefault(pid, []).append(r["rating"])

        if len(product_ratings) < 2:
            continue

        # Check pattern: one product all 5-star, others all 1-star
        avg_ratings = {
            pid: sum(ratings) / len(ratings)
            for pid, ratings in product_ratings.items()
        }
        high = [pid for pid, avg in avg_ratings.items() if avg >= 4.5]
        low = [pid for pid, avg in avg_ratings.items() if avg <= 1.5]

        if high and low:
            flags.append("account_manipulation")
            reasons.append(
                f"In category '{category}': boosting {len(high)} product(s) "
                f"(avg 5★) while tanking {len(low)} product(s) (avg 1★)."
            )
            break  # One detection is enough

    reasoning = " | ".join(reasons) if reasons else "Account passed cross-product check."

    logger.info(
        "Account check for %s — flags=%s", customer_id, flags,
    )

    return {
        "account_flags": flags,
        "reasoning": reasoning,
    }
