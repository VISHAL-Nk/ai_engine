"""
Router — Fake Detection (Microservice 1)

POST /api/fake-detection/analyze

Full pipeline: Preprocess → Bot Sniper → Sentiment → Trust Score → Bomb Detection
"""

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

from ai_engine.pipeline.preprocessor import preprocess_review
from ai_engine.pipeline.text_engine import analyze_text
from ai_engine.pipeline.vision_engine import analyze_image
from ai_engine.pipeline.fusion_layer import evaluate_fusion
from ai_engine.pipeline.bot_sniper import detect_bot_farm, check_account_spam
from ai_engine.pipeline.bomb_detector import detect_review_bomb

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/fake-detection", tags=["Fake Detection"])


# ── Request / Response schemas ───────────────────────────────────────────────


class AnalyzeRequest(BaseModel):
    review_id: str
    text: str
    rating: int = Field(ge=1, le=5)
    image_url: Optional[str] = None
    product_id: str
    customer_id: str


class AnalyzeResponse(BaseModel):
    review_id: str

    # Preprocessing
    cleaned_text: str
    detected_language: str
    preprocessing_flags: list[str]

    # Bot detection
    is_spam: bool
    is_duplicate: bool
    duplicate_of: Optional[str]
    similar_reviews: list[str]
    bot_reasoning: str

    # Account check
    account_flags: list[str]
    account_reasoning: str

    # Sentiment
    overall_sentiment: str
    sentiment_confidence: float
    image_classification: str
    is_sarcastic: bool

    # Trust
    trust_score: float
    risk_level: str
    is_incongruent: bool
    trust_reasoning: str

    # Bomb
    is_review_bomb: bool
    bomb_type: str
    bomb_reasoning: str

    # Final decision
    should_flag: bool
    flag_reasons: list[str]
    combined_reasoning: str


# ── Sarcasm heuristic ────────────────────────────────────────────────────────

import re

_SARCASM_PATTERNS = [
    re.compile(r"\boh\s+sure\b", re.IGNORECASE),
    re.compile(r"\byeah\s+right\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+a\s+joke\b", re.IGNORECASE),
    re.compile(r"\blove\s+how\b", re.IGNORECASE),
    re.compile(r"\bthanks\s+for\s+nothing\b", re.IGNORECASE),
]


def _detect_sarcasm(text: str, sentiment: str, rating: int) -> bool:
    if sentiment == "positive" and rating <= 2:
        return True
    if sentiment == "negative" and rating >= 5:
        return True
    for p in _SARCASM_PATTERNS:
        if p.search(text):
            return True
    return False


# ── Endpoint ─────────────────────────────────────────────────────────────────


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_review(request: AnalyzeRequest):
    """Run the full fake-detection pipeline on a single review."""

    logger.info("Analysing review %s …", request.review_id)

    # ── 1. Preprocess ────────────────────────────────────────────────────
    preprocess_result = preprocess_review(request.text)
    cleaned_text = preprocess_result["cleaned_text"]
    detected_language = preprocess_result["detected_language"]
    preprocessing_flags = preprocess_result["preprocessing_flags"]

    # ── 2. Bot Sniper (single-product) ───────────────────────────────────
    try:
        from ai_engine.db import get_recent_reviews_for_product
        recent = get_recent_reviews_for_product(request.product_id, limit=200)
        recent_for_bot = [
            {"_id": str(r["_id"]), "text": r.get("cleanedText") or r.get("text", "")}
            for r in recent
            if str(r.get("_id")) != request.review_id
            and (r.get("cleanedText") or r.get("text", "")).strip()
        ]
    except Exception as e:
        logger.warning("Could not fetch recent reviews from DB: %s", e)
        recent_for_bot = []

    bot_result = detect_bot_farm(cleaned_text, request.product_id, recent_for_bot)

    # ── 3. Cross-product account check ───────────────────────────────────
    try:
        from ai_engine.db import get_account_reviews
        account_reviews = get_account_reviews(request.customer_id)
    except Exception as e:
        logger.warning("Could not fetch account reviews from DB: %s", e)
        account_reviews = []

    try:
        account_result = check_account_spam(request.customer_id, account_reviews)
    except Exception as e:
        logger.exception("Account spam check failed for review %s: %s", request.review_id, e)
        account_result = {
            "account_flags": [],
            "reasoning": "Account-level spam analysis unavailable for this request.",
        }

    # ── 4. Sentiment analysis ────────────────────────────────────────────
    text_result = analyze_text(cleaned_text)
    sentiment = text_result["sentiment"]
    sentiment_confidence = text_result["confidence"]

    # Image analysis
    if request.image_url:
        try:
            image_result = analyze_image(request.image_url)
            image_classification = image_result.get("classification", "image_fetch_error")
        except Exception as e:
            logger.exception("Image analysis failed for review %s: %s", request.review_id, e)
            image_classification = "image_fetch_error"
    else:
        image_classification = "no_image"

    # Sarcasm check
    is_sarcastic = _detect_sarcasm(cleaned_text, sentiment, request.rating)

    # ── 5. Trust score (fusion) ──────────────────────────────────────────
    fusion_result = evaluate_fusion(sentiment, image_classification, request.rating)
    trust_score = fusion_result["trust_score"]
    risk_level = fusion_result["risk_level"]

    # ── 6. Review bomb detection ─────────────────────────────────────────
    try:
        from ai_engine.db import get_reviews_in_time_window
        recent_temporal = get_reviews_in_time_window(request.product_id, hours=24)
    except Exception as e:
        logger.warning("Could not fetch temporal reviews from DB: %s", e)
        recent_temporal = []

    try:
        bomb_result = detect_review_bomb(
            current_timestamp=datetime.now(timezone.utc),
            current_rating=request.rating,
            current_sentiment=sentiment,
            recent_reviews=recent_temporal,
        )
    except Exception as e:
        logger.exception("Bomb detection failed for review %s: %s", request.review_id, e)
        bomb_result = {
            "is_review_bomb": False,
            "bomb_type": "none",
            "reasoning": "Bomb detection unavailable for this request.",
        }

    # Penalise trust if bomb detected
    if bomb_result["is_review_bomb"]:
        trust_score = round(min(trust_score, 0.15), 2)
        risk_level = "high"

    # ── 7. Flag decision ─────────────────────────────────────────────────
    flag_reasons: list[str] = []

    if bot_result["is_bot"]:
        if bot_result["is_duplicate"]:
            flag_reasons.append("bot_duplicate")
        else:
            flag_reasons.append("promotional_spam")
    if account_result["account_flags"]:
        flag_reasons.extend(account_result["account_flags"])
    if trust_score <= 0.40:
        flag_reasons.append("high_risk")
    if bomb_result["is_review_bomb"]:
        flag_reasons.append("review_bomb")
    if is_sarcastic and sentiment_confidence < 0.60:
        flag_reasons.append("sarcastic_low_confidence")

    should_flag = len(flag_reasons) > 0

    # Combined reasoning
    reasoning_parts = []
    if bot_result["reasoning"] != "Review passed bot-farm similarity check.":
        reasoning_parts.append(bot_result["reasoning"])
    if account_result["account_flags"]:
        reasoning_parts.append(account_result["reasoning"])
    reasoning_parts.append(fusion_result["reasoning"])
    if bomb_result["is_review_bomb"]:
        reasoning_parts.append(bomb_result["reasoning"])
    if is_sarcastic:
        reasoning_parts.append(
            f"Sarcasm detected: '{sentiment}' sentiment with {request.rating}-star rating."
        )
    combined_reasoning = " | ".join(reasoning_parts)

    logger.info(
        "Review %s → sentiment=%s, trust=%.2f, flag=%s, reasons=%s",
        request.review_id, sentiment, trust_score, should_flag, flag_reasons,
    )

    return AnalyzeResponse(
        review_id=request.review_id,
        cleaned_text=cleaned_text,
        detected_language=detected_language,
        preprocessing_flags=preprocessing_flags,
        is_spam=bot_result["is_bot"],
        is_duplicate=bot_result["is_duplicate"],
        duplicate_of=bot_result["duplicate_of"],
        similar_reviews=bot_result["similar_reviews"],
        bot_reasoning=bot_result["reasoning"],
        account_flags=account_result["account_flags"],
        account_reasoning=account_result["reasoning"],
        overall_sentiment=sentiment,
        sentiment_confidence=sentiment_confidence,
        image_classification=image_classification,
        is_sarcastic=is_sarcastic,
        trust_score=trust_score,
        risk_level=risk_level,
        is_incongruent=fusion_result["is_incongruent"],
        trust_reasoning=fusion_result["reasoning"],
        is_review_bomb=bomb_result["is_review_bomb"],
        bomb_type=bomb_result["bomb_type"],
        bomb_reasoning=bomb_result["reasoning"],
        should_flag=should_flag,
        flag_reasons=flag_reasons,
        combined_reasoning=combined_reasoning,
    )
