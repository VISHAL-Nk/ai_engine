"""
Router — Tag Classification & Auto-Response (Microservice 3)

POST /api/tag-classification/generate      → Generate tags for a review
POST /api/tag-classification/auto-respond   → Generate auto-response for valid low-score review
"""

import logging
import os
import re
from typing import Optional

import requests as http_requests
from fastapi import APIRouter
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/tag-classification", tags=["Tag Classification"])

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral")


# ── Schemas ──────────────────────────────────────────────────────────────────


class TagRequest(BaseModel):
    review_id: str
    cleaned_text: str
    category: str
    overall_sentiment: str
    rating: int = Field(ge=1, le=5)


class FeatureSentimentResult(BaseModel):
    attribute: str
    sentiment: str
    confidence: float
    evidenceSnippet: str


class TagResponse(BaseModel):
    review_id: str
    tags: list[str]
    tag_count: int
    feature_sentiments: list[FeatureSentimentResult] = []


class AutoRespondRequest(BaseModel):
    review_id: str
    cleaned_text: str
    product_name: str
    category: str
    rating: int = Field(ge=1, le=5)
    tags: list[str]
    overall_sentiment: str


class AutoRespondResponse(BaseModel):
    review_id: str
    auto_response: str
    generated: bool


# ── LLM helper ───────────────────────────────────────────────────────────────


def _call_mistral(prompt: str, max_retries: int = 2) -> Optional[str]:
    """Call Mistral 7B via Ollama. Returns generated text or None on failure."""
    for attempt in range(max_retries):
        try:
            resp = http_requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": MISTRAL_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 200,
                    },
                },
                timeout=60,
            )
            resp.raise_for_status()
            return resp.json().get("response", "").strip()
        except Exception as e:
            logger.warning(
                "Mistral call failed (attempt %d/%d): %s",
                attempt + 1, max_retries, e,
            )
    return None


from ai_engine.pipeline import tag_generator as tg
from ai_engine.pipeline import auto_responder as ar

# ── Endpoints ────────────────────────────────────────────────────────────────


@router.post("/generate", response_model=TagResponse)
async def generate_tags(request: TagRequest):
    """Generate dynamic tags for a review using Mistral 7B."""
    
    tags = tg.generate_tags(
        text=request.cleaned_text,
        category=request.category,
        rating=request.rating,
        overall_sentiment=request.overall_sentiment,
        call_mistral_fn=_call_mistral
    )

    # ── Feature-level sentiment extraction ──────────────────────────────
    sentiment_list = tg.extract_feature_sentiments(
        request.cleaned_text, tags, request.overall_sentiment
    )
    
    feature_sentiments = [FeatureSentimentResult(**res) for res in sentiment_list]

    logger.info("Tags for review %s: %s (features: %d)", request.review_id, tags, len(feature_sentiments))

    return TagResponse(
        review_id=request.review_id,
        tags=tags,
        tag_count=len(tags),
        feature_sentiments=feature_sentiments,
    )


@router.post("/auto-respond", response_model=AutoRespondResponse)
async def auto_respond(request: AutoRespondRequest):
    """Generate an auto-response for a valid low-score review."""

    response_text, generated = ar.generate_auto_response(
        text=request.cleaned_text,
        product_name=request.product_name,
        category=request.category,
        rating=request.rating,
        tags=request.tags,
        call_mistral_fn=_call_mistral
    )

    logger.info("Auto-response for review %s: generated=%s", request.review_id, generated)

    return AutoRespondResponse(
        review_id=request.review_id,
        auto_response=response_text,
        generated=generated,
    )
