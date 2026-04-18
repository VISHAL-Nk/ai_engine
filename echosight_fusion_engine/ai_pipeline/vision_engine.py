"""
EchoSight Fusion Engine — Vision Classification (CLIP)
Uses OpenAI's CLIP model for zero-shot image classification
against a set of candidate labels relevant to product-review fraud.
"""

import logging
from io import BytesIO

import requests
from PIL import Image
from transformers import CLIPModel, CLIPProcessor  # type: ignore

logger = logging.getLogger(__name__)

# ── Model initialisation (runs once at import / first worker start) ──────────
MODEL_NAME = "openai/clip-vit-base-patch32"

logger.info("Loading CLIP vision model: %s …", MODEL_NAME)
_clip_model = CLIPModel.from_pretrained(MODEL_NAME)
_clip_processor = CLIPProcessor.from_pretrained(MODEL_NAME)
logger.info("CLIP model loaded successfully.")

# ── Candidate labels for zero-shot classification ───────────────────────────
CANDIDATE_LABELS: list[str] = [
    "a photo of a severely damaged product",
    "a photo of perfect intact packaging",
    "a screenshot of text or spam",
]

# ── Timeout for external image fetches (seconds) ────────────────────────────
_FETCH_TIMEOUT = 10


def analyze_image(image_url: str) -> str:
    """Fetch *image_url*, classify it with CLIP, and return the best label.

    Returns the candidate label with the highest probability, or
    ``"image_fetch_error"`` if the image could not be downloaded / decoded.
    """
    try:
        response = requests.get(image_url, timeout=_FETCH_TIMEOUT, stream=True)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
    except Exception:
        logger.exception("Failed to fetch or decode image from %s", image_url)
        return "image_fetch_error"

    # Run zero-shot classification
    inputs = _clip_processor(
        text=CANDIDATE_LABELS,
        images=image,
        return_tensors="pt",
        padding=True,
    )
    outputs = _clip_model(**inputs)

    # logits_per_image shape: (1, num_labels)
    logits = outputs.logits_per_image
    probs = logits.softmax(dim=1).detach().cpu().numpy()[0]

    best_idx = probs.argmax()
    best_label = CANDIDATE_LABELS[int(best_idx)]
    logger.debug(
        "CLIP classification: %s (prob=%.4f)",
        best_label,
        probs[int(best_idx)],
    )
    return best_label
