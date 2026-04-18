"""
EchoSight AI Engine — Vision Classification (CLIP)
Uses OpenAI CLIP for zero-shot image classification
against candidate labels relevant to product-review fraud.
"""

import logging
from io import BytesIO

import requests
from PIL import Image

logger = logging.getLogger(__name__)

# ── Lazy-loaded models ───────────────────────────────────────────────────────
MODEL_NAME = "openai/clip-vit-base-patch32"

_clip_model = None
_clip_processor = None


def _load_clip():
    global _clip_model, _clip_processor
    if _clip_model is None:
        from transformers import CLIPModel, CLIPProcessor  # type: ignore

        logger.info("Loading CLIP vision model: %s …", MODEL_NAME)
        _clip_model = CLIPModel.from_pretrained(MODEL_NAME)
        _clip_processor = CLIPProcessor.from_pretrained(MODEL_NAME)
        logger.info("CLIP model loaded successfully.")
    return _clip_model, _clip_processor


# ── Candidate labels ─────────────────────────────────────────────────────────
CANDIDATE_LABELS: list[str] = [
    "a photo of a severely damaged product",
    "a photo of perfect intact packaging",
    "a screenshot of text or spam",
]

_FETCH_TIMEOUT = 10


def analyze_image(image_url: str) -> dict:
    """Classify image with CLIP. Returns label + confidence.

    Returns:
        {
            "classification": str,
            "confidence": float,
            "all_scores": {label: float, ...}
        }
    """
    try:
        response = requests.get(image_url, timeout=_FETCH_TIMEOUT, stream=True)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
    except Exception:
        logger.exception("Failed to fetch or decode image from %s", image_url)
        return {
            "classification": "image_fetch_error",
            "confidence": 0.0,
            "all_scores": {},
        }

    model, processor = _load_clip()

    inputs = processor(
        text=CANDIDATE_LABELS,
        images=image,
        return_tensors="pt",
        padding=True,
    )
    outputs = model(**inputs)

    logits = outputs.logits_per_image
    probs = logits.softmax(dim=1).detach().cpu().numpy()[0]

    all_scores = {
        label: round(float(prob), 4)
        for label, prob in zip(CANDIDATE_LABELS, probs)
    }

    best_idx = probs.argmax()
    best_label = CANDIDATE_LABELS[int(best_idx)]
    best_conf = float(probs[int(best_idx)])

    logger.debug("CLIP classification: %s (prob=%.4f)", best_label, best_conf)

    return {
        "classification": best_label,
        "confidence": round(best_conf, 4),
        "all_scores": all_scores,
    }
