"""
EchoSight Fusion Engine — Pydantic Schemas
Defines the request/response contracts for the multimodal analysis API.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class ReviewRequest(BaseModel):
    """Incoming review payload for multimodal analysis."""

    review_id: str = Field(..., description="Unique identifier for the review.")
    text: str = Field(..., description="The textual content of the review.")
    image_url: Optional[str] = Field(
        None, description="URL of an image attached to the review, if any."
    )
    rating: int = Field(
        ..., ge=1, le=5, description="Star rating given by the user (1–5)."
    )
    review_timestamp: Optional[datetime] = Field(
        None,
        description=(
            "ISO-8601 timestamp of when the review was posted. "
            "Defaults to current server time if omitted."
        ),
    )


class FusionResult(BaseModel):
    """Result returned after running the multimodal fusion pipeline."""

    review_id: str = Field(..., description="Echo-back of the incoming review ID.")
    text_sentiment: str = Field(
        ..., description="Detected sentiment: positive, neutral, or negative."
    )
    image_classification: str = Field(
        ..., description="Best-matching CLIP candidate label or 'no_image'."
    )
    is_incongruent: bool = Field(
        ...,
        description="True when multimodal signals contradict each other.",
    )
    trust_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score (0–1) indicating review trustworthiness.",
    )
    is_review_bomb: bool = Field(
        ...,
        description=(
            "True when the review is part of a coordinated mass-review attack "
            "(volume spike + rating/sentiment clustering in a short time window)."
        ),
    )
    bomb_type: str = Field(
        ...,
        description=(
            "Type of bomb detected: 'negative_bomb', 'positive_bomb', "
            "'suspicious_cluster', or 'none'."
        ),
    )
    reasoning: str = Field(
        ..., description="Human-readable explanation of the fusion decision."
    )
