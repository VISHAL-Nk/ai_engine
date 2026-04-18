"""
EchoSight AI Engine — MongoDB connection (pymongo)

Read-only access for:
  - Fetching recent reviews (bot sniper)
  - Fetching account history (cross-product spam check)
  - Fetching temporal data (bomb detector)
"""

import os
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

from pymongo import MongoClient
from pymongo.database import Database

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency at runtime
    load_dotenv = None

logger = logging.getLogger(__name__)

_client: MongoClient | None = None
_db: Database | None = None
_env_bootstrapped = False


def _bootstrap_env() -> None:
    """Try to load local env files so MONGODB_URI is available in dev."""
    global _env_bootstrapped

    if _env_bootstrapped:
        return
    _env_bootstrapped = True

    if os.getenv("MONGODB_URI"):
        return

    if load_dotenv is None:
        logger.warning(
            "python-dotenv not available and MONGODB_URI is unset; "
            "history-based spam checks may be skipped."
        )
        return

    here = Path(__file__).resolve()
    repo_root = here.parents[1]
    cwd = Path.cwd().resolve()

    candidates = [
        repo_root / ".env",
        repo_root / ".env.local",
        repo_root / "web" / ".env.local",
        cwd / ".env",
        cwd / ".env.local",
        cwd / "web" / ".env.local",
    ]

    seen: set[str] = set()
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)

        if not path.exists():
            continue

        load_dotenv(dotenv_path=path, override=False)
        if os.getenv("MONGODB_URI"):
            logger.info("Loaded MONGODB_URI from %s", path)
            return

    logger.warning(
        "MONGODB_URI is not set and no local env file provided it; "
        "duplicate/bomb detection will have limited context."
    )


def get_db() -> Database:
    """Get the MongoDB database connection (lazy singleton)."""
    global _client, _db

    _bootstrap_env()

    if _db is None:
        uri = os.getenv("MONGODB_URI", "")
        if not uri:
            raise RuntimeError("MONGODB_URI environment variable not set.")
        _client = MongoClient(uri)

        try:
            _db = _client.get_default_database()
        except Exception:
            db_name = os.getenv("MONGODB_DB_NAME", "echosight")
            _db = _client[db_name]
            logger.warning(
                "MONGODB_URI did not include a default DB; using fallback DB '%s'.",
                db_name,
            )

        logger.info("Connected to MongoDB: %s", _db.name)
    return _db


def get_recent_reviews_for_product(product_id: str, limit: int = 100) -> list[dict]:
    """Fetch recent reviews for a product (for bot sniper)."""
    db = get_db()
    from bson import ObjectId

    cursor = db.reviews.find(
        {"productId": ObjectId(product_id)},
        {"_id": 1, "text": 1, "cleanedText": 1, "rating": 1, "createdAt": 1,
         "overallSentiment": 1, "customerId": 1},
    ).sort("createdAt", -1).limit(limit)

    return list(cursor)


def get_account_reviews(customer_id: str) -> list[dict]:
    """Fetch all reviews by a customer across all products (for account spam check)."""
    db = get_db()
    from bson import ObjectId

    pipeline = [
        {"$match": {"customerId": ObjectId(customer_id)}},
        {
            "$lookup": {
                "from": "products",
                "localField": "productId",
                "foreignField": "_id",
                "as": "product",
            }
        },
        {"$unwind": {"path": "$product", "preserveNullAndEmptyArrays": True}},
        {
            "$project": {
                "product_id": "$productId",
                "rating": 1,
                "text": 1,
                "created_at": "$createdAt",
                "category": "$product.category",
            }
        },
        {"$sort": {"created_at": -1}},
    ]

    return list(db.reviews.aggregate(pipeline))


def get_reviews_in_time_window(
    product_id: str,
    hours: int = 24,
) -> list[dict]:
    """Fetch reviews for a product within a time window (for bomb detector)."""
    db = get_db()
    from bson import ObjectId

    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

    cursor = db.reviews.find(
        {
            "productId": ObjectId(product_id),
            "createdAt": {"$gte": cutoff},
        },
        {"_id": 1, "rating": 1, "overallSentiment": 1, "createdAt": 1},
    ).sort("createdAt", -1)

    results = []
    for doc in cursor:
        results.append({
            "timestamp": doc.get("createdAt"),
            "rating": doc.get("rating", 3),
            "sentiment": doc.get("overallSentiment", "neutral"),
        })

    return results


def get_product_for_qa(product_id: str) -> dict | None:
    """Fetch active product metadata for customer Q&A."""
    db = get_db()
    from bson import ObjectId

    try:
        pid = ObjectId(product_id)
    except Exception:
        return None

    doc = db.products.find_one(
        {"_id": pid, "isActive": True},
        {
            "_id": 1,
            "name": 1,
            "description": 1,
            "category": 1,
            "price": 1,
            "isActive": 1,
        },
    )

    if not doc:
        return None

    return {
        "_id": str(doc.get("_id")),
        "name": doc.get("name", ""),
        "description": doc.get("description", ""),
        "category": doc.get("category", ""),
        "price": float(doc.get("price", 0) or 0),
        "isActive": bool(doc.get("isActive", False)),
    }


def get_approved_reviews_for_qa(product_id: str, limit: int = 40) -> list[dict]:
    """Fetch recent approved reviews with sentiment and feature snippets for Q&A."""
    db = get_db()
    from bson import ObjectId

    try:
        pid = ObjectId(product_id)
    except Exception:
        return []

    cursor = db.reviews.find(
        {
            "productId": pid,
            "moderationStatus": {"$in": ["approved", "auto_approved"]},
            "isFlagged": False,
        },
        {
            "_id": 1,
            "text": 1,
            "rating": 1,
            "overallSentiment": 1,
            "tags": 1,
            "featureSentiments": 1,
            "createdAt": 1,
        },
    ).sort("createdAt", -1).limit(limit)

    reviews = []
    for doc in cursor:
        reviews.append(
            {
                "_id": str(doc.get("_id")),
                "text": doc.get("text", ""),
                "rating": int(doc.get("rating", 0) or 0),
                "overallSentiment": doc.get("overallSentiment", "neutral"),
                "tags": doc.get("tags") or [],
                "featureSentiments": doc.get("featureSentiments") or [],
                "createdAt": doc.get("createdAt"),
            }
        )

    return reviews


def get_product_review_stats_for_qa(product_id: str) -> dict:
    """Compute compact aggregate stats for a product's approved reviews."""
    db = get_db()
    from bson import ObjectId

    try:
        pid = ObjectId(product_id)
    except Exception:
        return {
            "total_reviews": 0,
            "avg_rating": 0.0,
            "sentiment_breakdown": {"positive": 0, "neutral": 0, "negative": 0},
        }

    match_query = {
        "productId": pid,
        "moderationStatus": {"$in": ["approved", "auto_approved"]},
        "isFlagged": False,
    }

    stats_aggr = list(
        db.reviews.aggregate(
            [
                {"$match": match_query},
                {
                    "$group": {
                        "_id": None,
                        "total_reviews": {"$sum": 1},
                        "avg_rating": {"$avg": "$rating"},
                    }
                },
            ]
        )
    )

    sentiment_aggr = list(
        db.reviews.aggregate(
            [
                {"$match": match_query},
                {"$group": {"_id": "$overallSentiment", "count": {"$sum": 1}}},
            ]
        )
    )

    stats = stats_aggr[0] if stats_aggr else {}
    sentiments = {
        "positive": 0,
        "neutral": 0,
        "negative": 0,
    }
    for row in sentiment_aggr:
        key = str(row.get("_id") or "neutral").lower()
        if key in sentiments:
            sentiments[key] = int(row.get("count", 0) or 0)

    return {
        "total_reviews": int(stats.get("total_reviews", 0) or 0),
        "avg_rating": round(float(stats.get("avg_rating", 0) or 0), 2),
        "sentiment_breakdown": sentiments,
    }
