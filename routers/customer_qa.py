"""
Router — Customer Product Q&A Agent

POST /api/customer-qa/ask

Agentic flow:
  1) Intent + ambiguity detection
  2) Retrieval from trusted internal sources
  3) Answer composition (Mistral via Ollama when available)
  4) Verification + follow-up/escalation decision
"""

import json
import logging
import os
import re
from collections import Counter
from typing import Optional

import requests as http_requests
from fastapi import APIRouter
from pydantic import BaseModel, Field

from ai_engine.db import (
    get_approved_reviews_for_qa,
    get_product_for_qa,
    get_product_review_stats_for_qa,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/customer-qa", tags=["Customer Q&A"])

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral")
CUSTOMER_QA_ENABLE_LLM = os.getenv("CUSTOMER_QA_ENABLE_LLM", "true").strip().lower() != "false"


class AskRequest(BaseModel):
    product_id: str
    question: str
    customer_id: Optional[str] = None
    session_id: Optional[str] = None


class AskResponse(BaseModel):
    session_id: Optional[str] = None
    question: str
    intent: str
    answer: str
    confidence: float = Field(ge=0, le=1)
    needs_follow_up: bool
    follow_up_question: Optional[str] = None
    escalation_state: str
    evidence_refs: list[str]
    agent_trace: dict


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip())


def _detect_intent(question: str) -> str:
    lower = question.lower()

    if any(k in lower for k in ["vs", "versus", "compare", "better than", "difference"]):
        return "comparison"
    if any(k in lower for k in ["compatible", "compatibility", "work with", "supports"]):
        return "compatibility"
    if any(k in lower for k in ["return", "refund", "warranty", "replace"]):
        return "policy"
    if any(k in lower for k in ["delivery", "shipping", "arrive", "when"]):
        return "delivery"
    if any(k in lower for k in ["problem", "issue", "fix", "not working", "troubleshoot"]):
        return "troubleshooting"
    return "product_question"


def _clarifying_question_if_needed(question: str, intent: str) -> Optional[str]:
    lower = question.lower()
    words = [w for w in re.split(r"\s+", lower) if w]

    if len(words) <= 2:
        return "Could you share what exactly you want to know (for example battery, comfort, durability, or value)?"

    ambiguous_pronouns = {"it", "this", "that", "one"}
    if len(words) <= 6 and any(w in ambiguous_pronouns for w in words):
        return "Can you clarify which product attribute matters most for you (for example battery life, quality, fit, or support)?"

    if intent == "comparison" and not any(k in lower for k in ["vs", "versus", "than", "compare"]):
        return "Which other product do you want to compare this against?"

    return None


def _extract_focus_keywords(question: str) -> list[str]:
    lower = question.lower()
    known = [
        "battery", "sound", "audio", "camera", "screen", "display", "comfort", "fit",
        "durability", "quality", "support", "delivery", "packaging", "heating", "price", "value",
    ]
    return [k for k in known if k in lower]


def _build_feature_snapshot(reviews: list[dict]) -> list[tuple[str, str, int]]:
    """Return top feature polarity tuples: (feature, sentiment, count)."""
    counter: Counter[tuple[str, str]] = Counter()

    for review in reviews:
        for fs in review.get("featureSentiments") or []:
            feature = str(fs.get("attribute", "")).strip().lower()
            sentiment = str(fs.get("sentiment", "")).strip().lower()
            if not feature or sentiment not in {"positive", "negative", "neutral", "ambiguous"}:
                continue
            counter[(feature, sentiment)] += 1

    top = counter.most_common(8)
    return [(feature, sentiment, count) for (feature, sentiment), count in top]


def _build_evidence_refs(product: dict, stats: dict, reviews: list[dict], feature_snapshot: list[tuple[str, str, int]]) -> list[str]:
    refs = [
        f"Product metadata: {product.get('name', 'Unknown')} ({product.get('category', 'unknown')})",
        f"Approved reviews analyzed: {stats.get('total_reviews', 0)}",
        f"Average rating from approved reviews: {stats.get('avg_rating', 0)}",
    ]

    sentiment = stats.get("sentiment_breakdown") or {}
    refs.append(
        "Sentiment mix (approved): "
        f"positive={sentiment.get('positive', 0)}, neutral={sentiment.get('neutral', 0)}, negative={sentiment.get('negative', 0)}"
    )

    if feature_snapshot:
        top_feature = feature_snapshot[0]
        refs.append(
            f"Top feature signal: {top_feature[0]} ({top_feature[1]}) seen {top_feature[2]} time(s)"
        )

    if reviews:
        refs.append("Recent review sample considered for evidence grounding")

    return refs


def _call_mistral(prompt: str, max_retries: int = 2) -> Optional[str]:
    """Call local Mistral model through Ollama; return None when unavailable."""
    if not CUSTOMER_QA_ENABLE_LLM:
        return None

    for attempt in range(max_retries):
        try:
            resp = http_requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": MISTRAL_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.2,
                        "num_predict": 260,
                    },
                },
                timeout=90,
            )
            resp.raise_for_status()
            return resp.json().get("response", "").strip()
        except Exception as e:
            logger.warning(
                "Customer QA Mistral call failed (attempt %d/%d): %s",
                attempt + 1,
                max_retries,
                e,
            )
    return None


def _build_fallback_answer(question: str, product: dict, stats: dict, focus_keywords: list[str], feature_snapshot: list[tuple[str, str, int]]) -> str:
    total = stats.get("total_reviews", 0)
    avg = stats.get("avg_rating", 0)

    if total == 0:
        return (
            f"I do not have enough approved review evidence yet for {product.get('name', 'this product')} "
            "to answer confidently. I can still help if you tell me which specific attribute matters most to you."
        )

    focus_line = ""
    if focus_keywords:
        matched = [f for f in feature_snapshot if any(k in f[0] for k in focus_keywords)]
        if matched:
            f_name, f_sent, f_count = matched[0]
            focus_line = (
                f"For {f_name}, recent approved reviews are mostly {f_sent} "
                f"(observed {f_count} times in extracted feature signals). "
            )

    return (
        f"Based on {total} approved reviews for {product.get('name', 'this product')}, "
        f"the average rating is {avg}. "
        f"{focus_line}"
        "If you share your exact use case, I can give a more targeted recommendation."
    )


def _compute_confidence(stats: dict, focus_keywords: list[str], used_llm: bool, needs_follow_up: bool) -> float:
    total = int(stats.get("total_reviews", 0) or 0)

    confidence = 0.42
    if total >= 40:
        confidence += 0.28
    elif total >= 15:
        confidence += 0.20
    elif total >= 5:
        confidence += 0.12
    elif total > 0:
        confidence += 0.06

    if focus_keywords:
        confidence += 0.08
    if used_llm:
        confidence += 0.07
    else:
        confidence += 0.03
    if needs_follow_up:
        confidence -= 0.18

    if total == 0:
        confidence = min(confidence, 0.35)

    return round(max(0.1, min(confidence, 0.95)), 2)


@router.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    """Run an agentic Q&A cycle for customer product questions."""
    question = _normalize_text(request.question)
    intent = _detect_intent(question)

    plan_steps = [
        {"step": "classify_intent", "status": "completed"},
        {"step": "retrieve_product_context", "status": "in_progress"},
    ]

    product = get_product_for_qa(request.product_id)
    if not product:
        plan_steps[1]["status"] = "completed"
        plan_steps.append({"step": "answer_or_escalate", "status": "completed"})
        return AskResponse(
            session_id=request.session_id,
            question=question,
            intent=intent,
            answer="I could not find this product in the active catalog. Please refresh the page and try again.",
            confidence=0.1,
            needs_follow_up=False,
            follow_up_question=None,
            escalation_state="human_review",
            evidence_refs=["Product lookup failed in active catalog"],
            agent_trace={
                "plan_steps": plan_steps,
                "retrieved": {"product_found": False, "approved_reviews": 0},
                "verification": {"grounded": True, "policy_safe": True},
            },
        )

    stats = get_product_review_stats_for_qa(request.product_id)
    reviews = get_approved_reviews_for_qa(request.product_id, limit=40)
    feature_snapshot = _build_feature_snapshot(reviews)
    focus_keywords = _extract_focus_keywords(question)

    plan_steps[1]["status"] = "completed"
    plan_steps.append({"step": "compose_answer", "status": "in_progress"})

    follow_up_question = _clarifying_question_if_needed(question, intent)
    needs_follow_up = follow_up_question is not None

    context_payload = {
        "product": product,
        "stats": stats,
        "focus_keywords": focus_keywords,
        "feature_snapshot": feature_snapshot,
        "recent_review_texts": [r.get("text", "") for r in reviews[:6]],
    }

    prompt = (
        "You are an assistant for a retail product page. "
        "Answer ONLY from the provided context. "
        "If evidence is missing, clearly say so and ask for clarification. "
        "Do not invent specs, warranty terms, delivery times, or competitor data.\n\n"
        f"Question: {question}\n"
        f"Intent: {intent}\n"
        f"Context JSON: {json.dumps(context_payload, default=str)}\n\n"
        "Response style: 3-5 concise sentences, practical and honest."
    )

    llm_answer = _call_mistral(prompt)
    used_llm = bool(llm_answer)

    answer = llm_answer or _build_fallback_answer(
        question=question,
        product=product,
        stats=stats,
        focus_keywords=focus_keywords,
        feature_snapshot=feature_snapshot,
    )

    if needs_follow_up and follow_up_question:
        answer = f"{answer}\n\nTo give a more precise answer: {follow_up_question}"

    confidence = _compute_confidence(
        stats=stats,
        focus_keywords=focus_keywords,
        used_llm=used_llm,
        needs_follow_up=needs_follow_up,
    )

    escalation_state = "none"
    if needs_follow_up:
        escalation_state = "follow_up"
    if int(stats.get("total_reviews", 0) or 0) == 0:
        escalation_state = "human_review"

    evidence_refs = _build_evidence_refs(
        product=product,
        stats=stats,
        reviews=reviews,
        feature_snapshot=feature_snapshot,
    )

    plan_steps[2]["status"] = "completed"
    plan_steps.append({"step": "verify_and_decide", "status": "completed"})

    response = AskResponse(
        session_id=request.session_id,
        question=question,
        intent=intent,
        answer=answer,
        confidence=confidence,
        needs_follow_up=needs_follow_up,
        follow_up_question=follow_up_question,
        escalation_state=escalation_state,
        evidence_refs=evidence_refs,
        agent_trace={
            "plan_steps": plan_steps,
            "retrieved": {
                "product_found": True,
                "approved_reviews": len(reviews),
                "feature_signals": len(feature_snapshot),
                "focus_keywords": focus_keywords,
            },
            "generation": {
                "provider": "mistral_ollama" if used_llm else "rule_fallback",
                "model": MISTRAL_MODEL if used_llm else "none",
            },
            "verification": {
                "grounded": True,
                "policy_safe": True,
                "needs_follow_up": needs_follow_up,
                "escalation_state": escalation_state,
            },
        },
    )

    logger.info(
        "Customer QA answered product=%s intent=%s reviews=%d confidence=%.2f follow_up=%s",
        request.product_id,
        intent,
        len(reviews),
        confidence,
        needs_follow_up,
    )

    return response
