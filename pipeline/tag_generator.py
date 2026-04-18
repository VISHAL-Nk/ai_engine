"""
Tag Generator Module
Extracts keyword-based or LLM-based descriptive tags from product reviews.
"""
import re
import os
import logging
import requests as http_requests

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral")

_POSITIVE_WORDS = {"great", "good", "amazing", "excellent", "love", "best", "perfect", "stunning", "smooth", "fast", "long", "beautiful", "comfortable", "fresh", "delicious", "soft"}
_NEGATIVE_WORDS = {"bad", "poor", "terrible", "worst", "hate", "broken", "slow", "warm", "hot", "heavy", "cheap", "stale", "rough", "tight", "loose", "fading", "issue", "problem"}

def _extract_tags_fallback(text: str, category: str) -> list[str]:
    text_lower = text.lower()
    tags: list[str] = []

    keyword_map = {
        "electronics": [
            "battery", "screen", "camera", "charging", "software", "performance",
            "build quality", "sound", "speaker", "display", "lag", "heating",
            "fingerprint", "face unlock", "storage", "ram",
        ],
        "food": [
            "taste", "freshness", "packaging", "delivery", "expiry", "smell",
            "quantity", "ingredients", "flavour", "stale", "aroma", "spicy",
        ],
        "clothing": [
            "fabric", "colour", "color", "stitching", "fit", "comfort",
            "size", "design", "wash", "fading", "shrink", "tear", "zipper",
        ],
    }

    keywords = keyword_map.get(category, keyword_map.get("electronics", []))

    for kw in keywords:
        if kw in text_lower:
            pattern = re.compile(rf"(\b\w+\s+)?{re.escape(kw)}(\s+\w+\b)?", re.IGNORECASE)
            match = pattern.search(text)
            if match:
                tag = match.group(0).strip().lower()
                if len(tag.split()) <= 4 and tag not in tags:
                    tags.append(tag)
            elif kw not in tags:
                tags.append(kw)

    return tags[:5]

def extract_feature_sentiments(text: str, tags: list[str], overall_sentiment: str) -> list[dict]:
    results = []
    text_lower = text.lower()
    sentences = re.split(r'[.!?]+', text)

    for tag in tags:
        tag_lower = tag.lower()
        evidence = ""
        for sent in sentences:
            if tag_lower in sent.lower():
                evidence = sent.strip()
                break

        if not evidence:
            evidence = text[:100]

        evidence_lower = evidence.lower()
        pos_count = sum(1 for w in _POSITIVE_WORDS if w in evidence_lower)
        neg_count = sum(1 for w in _NEGATIVE_WORDS if w in evidence_lower)

        if pos_count > neg_count:
            sentiment = "positive"
            confidence = min(0.6 + pos_count * 0.1, 0.95)
        elif neg_count > pos_count:
            sentiment = "negative"
            confidence = min(0.6 + neg_count * 0.1, 0.95)
        else:
            sentiment = overall_sentiment if overall_sentiment in ("positive", "negative") else "neutral"
            confidence = 0.5

        results.append({
            "attribute": tag,
            "sentiment": sentiment,
            "confidence": round(confidence, 2),
            "evidenceSnippet": evidence[:200],
        })

    return results

def generate_tags(text: str, category: str, rating: int, overall_sentiment: str, call_mistral_fn) -> list[str]:
    prompt = (
        "Extract 3-5 short descriptive tags from this product review. "
        "Tags should describe the specific issues or praises mentioned. "
        "Return ONLY the tags as a comma-separated list, nothing else.\n\n"
        f"Product category: {category}\n"
        f"Rating: {rating}/5\n"
        f"Sentiment: {overall_sentiment}\n"
        f"Review: \"{text}\"\n\n"
        "Tags:"
    )

    result = call_mistral_fn(prompt)

    if result:
        raw_tags = [t.strip().lower().strip('"\'') for t in result.split(",")]
        tags = [t for t in raw_tags if 1 < len(t) < 50 and t][:5]
    else:
        logger.info("Mistral unavailable — using fallback config.")
        tags = _extract_tags_fallback(text, category)

    if not tags:
        tags = _extract_tags_fallback(text, category)
        
    return tags
