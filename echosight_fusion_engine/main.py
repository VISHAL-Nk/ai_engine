"""
EchoSight Fusion Engine — FastAPI Application
Exposes a single POST endpoint that orchestrates the full
multimodal analysis pipeline: NLP → Vision → Fusion → Bomb Check.
"""

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException

from echosight_fusion_engine.models import FusionResult, ReviewRequest
from echosight_fusion_engine.ai_pipeline.text_engine import analyze_text
from echosight_fusion_engine.ai_pipeline.vision_engine import analyze_image
from echosight_fusion_engine.ai_pipeline.fusion_layer import evaluate_fusion
from echosight_fusion_engine.ai_pipeline.bot_sniper import detect_bot_farm
from echosight_fusion_engine.ai_pipeline.review_bomb_detector import detect_review_bomb

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ── Lifespan event (models are loaded at import time, but we log readiness) ─
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 EchoSight Fusion Engine is ready.")
    yield
    logger.info("🛑 EchoSight Fusion Engine shutting down.")


# ── App instance ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="EchoSight Fusion Engine",
    description=(
        "Multimodal AI microservice that cross-references textual review "
        "sentiment with image evidence to detect fraudulent or incongruent "
        "product reviews."
    ),
    version="2.0.0",
    lifespan=lifespan,
)


# ── Health check ─────────────────────────────────────────────────────────────
@app.get("/health", tags=["ops"])
async def health_check():
    return {"status": "healthy"}


# ── Core endpoint ────────────────────────────────────────────────────────────
@app.post(
    "/api/analyze-multimodal",
    response_model=FusionResult,
    tags=["analysis"],
    summary="Run multimodal review analysis",
    description=(
        "Accepts a product review (text + optional image URL + rating + timestamp), "
        "runs NLP sentiment analysis, CLIP-based image classification, "
        "weighted trust scoring, and temporal review-bomb detection."
    ),
)
async def analyze_multimodal(request: ReviewRequest) -> FusionResult:
    """Orchestrate the full NLP → Vision → Fusion → Bomb pipeline."""

    logger.info("Processing review %s …", request.review_id)

    # ── 0. Bot-farm detection (fast TF-IDF check before heavy inference) ─
    recent_db_reviews: list[str] = []  # TODO: wire to Supabase
    bot_check = detect_bot_farm(request.text, "prod-current", recent_db_reviews)

    if bot_check["is_bot"]:
        logger.warning(
            "Review %s flagged as bot — short-circuiting pipeline.",
            request.review_id,
        )
        return FusionResult(
            review_id=request.review_id,
            text_sentiment="neutral",
            image_classification="no_image",
            is_incongruent=False,
            trust_score=0.0,
            is_review_bomb=False,
            bomb_type="none",
            reasoning=bot_check["reasoning"],
        )

    # ── 1. Text sentiment analysis ───────────────────────────────────────
    try:
        text_sentiment = analyze_text(request.text)
    except Exception as exc:
        logger.exception("Text analysis failed for review %s", request.review_id)
        raise HTTPException(
            status_code=500,
            detail=f"Text analysis failed: {exc}",
        ) from exc

    # ── 2. Image classification (if an image URL is provided) ────────────
    if request.image_url:
        try:
            image_classification = analyze_image(request.image_url)
        except Exception as exc:
            logger.exception("Image analysis failed for review %s", request.review_id)
            raise HTTPException(
                status_code=500,
                detail=f"Image analysis failed: {exc}",
            ) from exc
    else:
        image_classification = "no_image"

    # ── 3. Weighted trust score (text ↔ rating ↔ image alignment) ────────
    fusion = evaluate_fusion(text_sentiment, image_classification, request.rating)

    # ── 4. Review-bomb detection (temporal pattern analysis) ─────────────
    timestamp = request.review_timestamp or datetime.now(timezone.utc)

    # TODO: replace with actual Supabase query for recent reviews
    recent_product_reviews: list[dict] = []

    bomb_result = detect_review_bomb(
        current_timestamp=timestamp,
        current_rating=request.rating,
        current_sentiment=text_sentiment,
        recent_reviews=recent_product_reviews,
    )

    # If review bomb detected, penalise trust score
    final_trust = fusion["trust_score"]
    if bomb_result["is_review_bomb"]:
        final_trust = round(min(final_trust, 0.15), 2)

    # ── 5. Assemble response ─────────────────────────────────────────────
    # Combine reasoning from fusion + bomb detector
    combined_reasoning = fusion["reasoning"]
    if bomb_result["is_review_bomb"]:
        combined_reasoning = bomb_result["reasoning"] + " | " + combined_reasoning

    result = FusionResult(
        review_id=request.review_id,
        text_sentiment=text_sentiment,
        image_classification=image_classification,
        is_incongruent=fusion["is_incongruent"],
        trust_score=final_trust,
        is_review_bomb=bomb_result["is_review_bomb"],
        bomb_type=bomb_result["bomb_type"],
        reasoning=combined_reasoning,
    )

    logger.info(
        "Review %s → sentiment=%s, image=%s, trust=%.2f, bomb=%s",
        request.review_id,
        result.text_sentiment,
        result.image_classification,
        result.trust_score,
        result.is_review_bomb,
    )
    return result
