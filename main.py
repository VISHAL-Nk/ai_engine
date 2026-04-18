"""
EchoSight AI Engine — FastAPI Application

A single FastAPI app with 4 logical microservice routers:
  1. /api/fake-detection   → Preprocessing + Bot + Sentiment + Trust + Bomb
  2. /api/trend-analysis   → Time-series trend detection
  3. /api/tag-classification → Tag generation + Auto-response
    4. /api/customer-qa      → Agentic customer product Q&A
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from ai_engine.routers import fake_detection, trend_analysis, tag_classification, customer_qa

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ── Lifespan ─────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 EchoSight AI Engine starting up…")
    yield
    logger.info("🛑 EchoSight AI Engine shutting down.")


# ── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="EchoSight AI Engine",
    description=(
        "AI microservice for the Customer Review Intelligence Platform. "
        "Handles fake detection, trend analysis, tag classification, and customer Q&A."
    ),
    version="3.0.0",
    lifespan=lifespan,
)

# ── Register routers ────────────────────────────────────────────────────────
app.include_router(fake_detection.router)
app.include_router(trend_analysis.router)
app.include_router(tag_classification.router)
app.include_router(customer_qa.router)


# ── Health check ─────────────────────────────────────────────────────────────
@app.get("/health", tags=["ops"])
async def health_check():
    return {"status": "healthy", "service": "echosight-ai-engine"}
