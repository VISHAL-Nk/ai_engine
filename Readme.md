# EchoSight AI Engine

EchoSight AI Engine is a FastAPI microservice that powers fraud detection, trend detection, tag generation, auto-response generation, and customer product Q&A.

## Core Services

- Fake Detection:
  - preprocessing
  - bot and duplicate detection
  - account-level spam checks
  - sentiment and image analysis
  - trust scoring and bomb detection
- Trend Analysis:
  - sliding-window feature trend analysis
  - trend snapshot persistence
  - seller/admin alert generation
- Tag Classification:
  - review tag extraction
  - feature-level sentiment extraction
  - LLM-assisted when available, rule fallback otherwise
- Customer Q&A:
  - intent detection
  - retrieval from trusted internal product/review data
  - grounded answer generation with fallback logic

## Active Entrypoint

- Main service entrypoint: ai_engine/main.py
- Router package: ai_engine/routers
- Pipeline modules: ai_engine/pipeline
- MongoDB helpers: ai_engine/db.py
- Legacy fusion-only prototype remains under ai_engine/echosight_fusion_engine

## Tech Stack

- FastAPI + Pydantic
- Transformers + Torch
- scikit-learn
- Pillow
- pymongo
- Optional local Ollama Mistral integration for generation tasks

## Environment Variables

Create an .env file in the ai_engine module (or provide via environment):

| Variable | Required | Description | Example |
|---|---|---|---|
| MONGODB_URI | Yes | MongoDB connection string | mongodb+srv://... |
| MONGODB_DB_NAME | No | Fallback DB name if URI has no default DB | echosight |
| OLLAMA_BASE_URL | No | Ollama base URL | http://localhost:11434 |
| MISTRAL_MODEL | No | Ollama model name | mistral |
| CUSTOMER_QA_ENABLE_LLM | No | Enable LLM generation in Q&A | true |

## Installation

1. Create and activate a Python virtual environment.
2. Install dependencies from requirements.txt in the ai_engine module.
3. Configure environment variables.

Example commands:

python -m venv .venv  
source .venv/bin/activate  
pip install -r requirements.txt

## Run the Service

Run from the repository root so package imports resolve correctly:

uvicorn ai_engine.main:app --host 0.0.0.0 --port 8000 --reload

## Health and Docs

- Health: GET /health
- Swagger UI: /docs
- OpenAPI JSON: /openapi.json

## API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| POST | /api/fake-detection/analyze | Full review fraud analysis pipeline |
| POST | /api/trend-analysis/detect | Run trend detection for a product |
| GET | /api/trend-analysis/timeline/:product_id | Return feature timeline data |
| POST | /api/tag-classification/generate | Generate tags and feature sentiments |
| POST | /api/tag-classification/auto-respond | Generate customer support auto-response |
| POST | /api/customer-qa/ask | Agentic customer product Q&A |

## Fake Detection Pipeline Summary

1. Preprocess text:
   - language detection
   - optional translation
   - emoji normalization
   - typo correction
2. Detect bot and duplicate behavior:
   - exact normalized duplicate checks
   - TF-IDF similarity checks
   - promotional spam pattern checks
3. Account-level checks:
   - high review velocity
   - cross-product similarity
   - rating manipulation patterns
4. Sentiment and optional image analysis.
5. Trust fusion scoring.
6. Review bomb detection.
7. Final flag decision with explainable reasoning.

## Degradation Behavior

- If MongoDB context fetch fails, pipeline continues with limited historical context.
- If image analysis fails, result is downgraded to image_fetch_error instead of hard-failing.
- If Ollama/Mistral is unavailable:
  - tag generation uses deterministic fallback logic
  - auto-response uses template response
  - customer Q&A uses grounded fallback answer

## Integration with Web Module

- Web sends review analysis requests to:
  - /api/fake-detection/analyze
  - /api/tag-classification/generate
  - /api/tag-classification/auto-respond
  - /api/customer-qa/ask
- Set AI_ENGINE_URL in web configuration to this service URL.

## Recommended Startup Order

1. Start MongoDB connectivity.
2. Start AI engine.
3. Start web module.
4. Verify /health in both services.