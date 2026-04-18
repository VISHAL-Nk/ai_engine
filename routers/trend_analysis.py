"""
Router — Trend Analysis (Microservice 2)

POST /api/trend-analysis/detect     → Run trend detection for a product
GET  /api/trend-analysis/timeline   → Get feature timeline data for charts
"""

import logging
from collections import defaultdict

from fastapi import APIRouter, Query
from pydantic import BaseModel
from typing import Optional

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/trend-analysis", tags=["Trend Analysis"])


# ── Schemas ──────────────────────────────────────────────────────────────────


class TrendAlert(BaseModel):
    feature: str
    direction: str
    severity: str
    current_ratio: float
    previous_ratio: float
    change_pct: float
    window_description: str
    unique_reviewers: int
    example_reviews: list[str]


class TrendDetectResponse(BaseModel):
    product_id: str
    total_reviews_analyzed: int
    trend_alerts: list[TrendAlert]
    feature_timeline: dict[str, list[dict]]


# ── Endpoint ─────────────────────────────────────────────────────────────────


@router.post("/detect", response_model=TrendDetectResponse)
async def detect_trends(product_id: str, window_size: int = 50):
    """Run sliding-window trend detection for a product."""
    from ai_engine.db import get_db
    from bson import ObjectId
    from ai_engine.pipeline import trend_detector
    from datetime import datetime

    db = get_db()
    product = None
    reviews = []
    
    try:
        pid = ObjectId(product_id)
        product = db.products.find_one({"_id": pid})
        reviews = list(
            db.reviews.find(
                {
                    "productId": pid,
                    "moderationStatus": {"$in": ["approved", "auto_approved"]},
                    "isFlagged": False,
                },
                {
                    "featureSentiments": 1,
                    "customerId": 1,
                    "text": 1,
                    "createdAt": 1,
                },
            ).sort("createdAt", 1)
        )
    except Exception as e:
        logger.error("Failed to fetch product/reviews for trend analysis: %s", e)

    total = len(reviews)
    if total < window_size or not product:
        return TrendDetectResponse(
            product_id=product_id,
            total_reviews_analyzed=total,
            trend_alerts=[],
            feature_timeline={},
        )

    # 1. Build sliding windows
    windows = trend_detector.compute_windows(reviews, window_size)
    if len(windows) < 2:
        return TrendDetectResponse(
            product_id=product_id,
            total_reviews_analyzed=total,
            trend_alerts=[],
            feature_timeline={},
        )

    # 2. Compute timeline
    feature_timeline_dict = trend_detector.compute_feature_timeline(windows)

    # 3. Analyze trends
    alerts_raw = trend_detector.analyze_trends(windows, feature_timeline_dict, window_size)
    
    trend_alerts = [TrendAlert(**a) for a in alerts_raw]

    # 4. Persistence & Notifications
    try:
        # A. Save Snapshot to Trends collection
        doc_snapshot = {
            "productId": pid,
            "periodStart": windows[0][0].get("createdAt"),
            "periodEnd": windows[-1][-1].get("createdAt"),
            "totalReviewsAnalyzed": total,
            "windowSize": window_size,
            "featureTimeline": feature_timeline_dict,
            "detectedTrends": alerts_raw,
            "createdAt": datetime.utcnow()
        }
        db.trends.insert_one(doc_snapshot)

        # B. Generate Alerts if severity is not 'isolated'
        for a in alerts_raw:
            if a.get("severity") in ("emerging", "systemic"):
                msg = f"{a['direction'].replace('_', ' ').capitalize()} for '{a['feature']}'. Ratio changed by {a['change_pct']}%. Affected {a['unique_reviewers']} unique reviewers in latest {window_size} reviews."

                # Try to group by checking existing unread alert
                existing_seller = db.alerts.find_one({
                    "recipientRole": "seller",
                    "recipientId": product.get("sellerId"),
                    "type": "trend_alert",
                    "title": {"$regex": a['feature']},
                    "relatedProductId": pid,
                    "isRead": False,
                })

                if not existing_seller:
                    alert_doc = {
                        "recipientRole": "seller",
                        "recipientId": product.get("sellerId"),
                        "type": "trend_alert",
                        "title": f"Trend Alert: {a['feature'].capitalize()}",
                        "message": msg,
                        "relatedProductId": pid,
                        "relatedReviewIds": [], 
                        "isRead": False,
                        "createdAt": datetime.utcnow(),
                        "updatedAt": datetime.utcnow()
                    }
                    db.alerts.insert_one(alert_doc)
                
                # Admin alert for systemic issues
                if a.get("severity") == "systemic":
                    existing_admin = db.alerts.find_one({
                        "recipientRole": "admin",
                        "type": "trend_alert",
                        "title": {"$regex": a['feature']},
                        "relatedProductId": pid,
                        "isRead": False,
                    })
                    if not existing_admin:
                        db.alerts.insert_one({
                            "recipientRole": "admin",
                            "type": "trend_alert",
                            "title": f"Systemic Issue on {product.get('name')[:15]}",
                            "message": msg,
                            "relatedProductId": pid,
                            "relatedReviewIds": [], 
                            "isRead": False,
                            "createdAt": datetime.utcnow(),
                            "updatedAt": datetime.utcnow()
                        })

    except Exception as db_err:
        logger.error(f"Failed to persist trends or emit alerts: {db_err}")

    logger.info("Trend analysis for product %s — %d alerts", product_id, len(trend_alerts))

    return TrendDetectResponse(
        product_id=product_id,
        total_reviews_analyzed=total,
        trend_alerts=trend_alerts,
        feature_timeline=dict(feature_timeline_dict),
    )


@router.get("/timeline/{product_id}")
async def get_timeline(product_id: str, window_size: int = Query(default=50)):
    """Get feature timeline data for charting (seller dashboard)."""
    result = await detect_trends(product_id, window_size)
    return {
        "product_id": result.product_id,
        "feature_timeline": result.feature_timeline,
    }
