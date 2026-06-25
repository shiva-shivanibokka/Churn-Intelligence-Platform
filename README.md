# Customer Segmentation & Churn Engine

> End-to-end decision intelligence platform: behavioral segmentation → per-cohort churn prediction → uplift modeling → 12-tool AI retention agent → closed-loop outcome tracking.

![CI](https://github.com/shiva-shivanibokka/Churn-Intelligence-Platform/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.12-blue)
![Docker](https://img.shields.io/badge/docker-build-passing)

---

## Recruiter TL;DR

- **What it does:** Segments customers behaviorally, predicts churn per cohort with calibrated probabilities, identifies the subset worth spending retention budget on (uplift modeling), and deploys a 12-tool ReAct AI agent that reasons over SHAP drivers, intervention history, and ROI before generating and saving a personalized retention plan.
- **Hardest problem solved:** Replacing a naive "email everyone above 0.7 churn probability" approach with causal uplift modeling (CausalML T-Learner + S-Learner) to distinguish Persuadables from Lost Causes and Sleeping Dogs — the same targeting logic Uber open-sourced CausalML to solve.
- **Verified results from pipeline runs:** CV AUC 0.945–0.984 across 5 segments; cluster stability mean ARI = 0.921 across 100 bootstrap resamplings; 710 Persuadables identified from 948 high-risk customers on the e-commerce dataset.

---

## Why This Exists

Most churn projects do: `features → model → churn probability → send email to everyone above 0.7`

That approach has two compounding problems. First, a single global model treats Champions and Lapsed customers identically — but a Champion churns for a specific trigger (bad support experience, competitor offer) while a Lapsed customer churns through gradual disengagement. Second, churn probability alone is the wrong optimization target: you should be targeting customers who will *respond* to intervention, not just customers who will churn.

This project was built to demonstrate what a production retention system actually looks like — the architecture used by Salesforce, Uber, and Netflix — end to end, with real artifacts: a serving API, a CI pipeline, Docker containerization, a full dashboard, and a closed-loop feedback system.

---

## What the Full System Does

```
Raw behavioral data (3 supported datasets)
  → Schema validation + median imputation
  → 8 engineered composite features
  → K-Means++ segmentation + GMM soft probability assignments
  → Bootstrap ARI stability validation (100 resamplings)
  → Per-segment CatBoost classifiers + isotonic calibration
  → Gain-based feature importance (per-customer SHAP approximation)
  → CausalML T-Learner + S-Learner uplift modeling
  → Four-quadrant customer classification (Persuadable / Sure Thing / Lost Cause / Sleeping Dog)
  → Intervention ROI ranking (uplift × CLV − cost)
  → Supabase (PostgreSQL) — 5 tables, 10 RPCs
  → Next.js 16 dashboard (5 pages, Recharts + Plotly)
  → 12-tool ReAct AI agent (Groq llama-3.3-70b-versatile, max 5 rounds)
  → Retention action audit trail + CSM outcome feedback loop
```

---

## Architecture

```mermaid
flowchart TD
    subgraph pipeline["Python ML Pipeline"]
        A[Raw Data] --> B[Feature Engineering]
        B --> C["Segmentation<br/>K-Means++ · GMM · PaCMAP"]
        C --> D["Churn Prediction<br/>Per-segment CatBoost"]
        D --> E["Uplift Modeling<br/>T-Learner · S-Learner"]
    end

    subgraph api["FastAPI Serving"]
        F["/health · /readiness · /score"]
    end

    E -->|enriched customers table| G[("Supabase PostgreSQL<br/>5 tables · 10 RPCs")]
    E --> F

    subgraph dashboard["Next.js 16 Dashboard"]
        H[Segmentation page]
        I[Churn page]
        J[Uplift page]
        K[Retention page]
        L[Analytics page]
    end

    G --> dashboard

    subgraph agent["ReAct AI Agent"]
        M["llama-3.3-70b-versatile<br/>12 tools · max 5 rounds"]
    end

    K --> agent
    agent -->|saves retention_actions| G
    L -->|reads retention_actions + feedback| G
```

**Why it's shaped this way:**

- **Per-segment models over a global model.** A Champion and a Lapsed customer churn for fundamentally different reasons. Separate CatBoost classifiers per cohort capture segment-specific dynamics. This mirrors Salesforce Einstein's per-tier health scoring.
- **Isotonic calibration over raw probabilities.** Raw CatBoost output is not well-calibrated — a score of 0.7 ≠ 70% actual churn rate. Calibration is required whenever probabilities drive financial calculations (CLV, retention ROI, budget allocation). Isotonic regression is preferred over Platt scaling for non-parametric distributions.
- **Observational uplift instead of A/B targeting.** The datasets don't include historical experiment logs. Treatment proxies (`Complain` flag = received support outreach; `CouponUsed > 0` = received discount) follow the academic literature on observational uplift. Production systems (Uber, Netflix) train on actual randomized experiment logs.
- **Server-side retention action saves.** The `retention_actions` table has Row Level Security enabled in Supabase. The AI agent API route uses the service role key (server-side only) for inserts — the browser anon key is read-only.
- **DB-driven agent configuration.** The system prompt is rebuilt from the `business_config` table on every request. Changing CLV assumptions, intervention types, or outreach channels requires only a database row update — no code change, no redeploy.

---

## Features

**ML Pipeline**
- 8 engineered composite features: `EngagementScore`, `RecencySignal`, `StickinessIndex`, `SpendTrend`, `SupportRiskScore`, `DiscountSensitivity`, `TenureStability`, `WarehouseFriction`
- Schema validation with column presence and missing-rate checks before any transformation
- Supports 3 datasets selectable via `--dataset` CLI flag: e-commerce (5,630 customers), Olist Brazilian marketplace (42,325), Cell2Cell telecom (~71,000)
- K-Means++ with 5 clusters + GMM soft probability assignments (each customer gets a probability distribution across segments, not just a hard label)
- Bootstrap cluster stability: Adjusted Rand Index across 100 resamplings with a pass/warn/fail grading scheme
- Per-segment CatBoost classifiers with stratified 80/20 holdout and 5-fold cross-validation
- Isotonic probability calibration for reliable downstream ROI calculations
- MLflow experiment tracking — one run logged per segment per training
- CausalML T-Learner + S-Learner uplift models with four-quadrant customer classification
- Intervention ROI ranking: `net_roi = uplift_score × CLV − intervention_cost`

**FastAPI Scoring Endpoint**
- `GET /health` — liveness probe
- `GET /readiness` — readiness probe (confirms model artifacts loaded)
- `POST /score` — accepts raw customer features, returns `segment`, `churn_probability`, `risk_tier`, `customer_type`
- Input validated with Pydantic; returns 422 on missing or out-of-range fields

**Next.js Dashboard (5 pages)**
- **Segmentation** — segment heatmap, PaCMAP behavioral scatter colored by segment, GMM soft probability heatmap
- **Churn** — KPI cards, churn probability histogram, SHAP-based feature importance bar chart, risk tier breakdown by segment, average churn by segment
- **Uplift** — customer type funnel, ROI by segment, top Persuadable priority list, uplift vs. churn probability scatter
- **Retention** — Persuadable customer list, AI agent in two modes (batch auto-generate or conversational chat), collapsible agent reasoning trace
- **Analytics** — full audit log of generated actions, outcome feedback (retained / churned / pending), success rate by intervention type

**AI Agent (12 tools)**

| Tool | Purpose |
|---|---|
| `get_top_churn_drivers` | SHAP-approximated churn drivers per customer |
| `get_segment_benchmark` | Average metrics for a named segment |
| `calculate_intervention_roi` | Net ROI given uplift, CLV, and cost |
| `lookup_customer_details` | Full customer record by ID |
| `search_retention_playbook` | DB-driven playbook lookup by risk factor keyword |
| `get_all_segment_benchmarks` | Cross-segment comparison in one call |
| `get_past_interventions` | Intervention history per customer |
| `get_intervention_success_rates` | Historical retention rates by intervention type |
| `get_at_risk_customers` | Top high-risk customers, optionally by segment |
| `get_revenue_at_risk` | Expected churner count × CLV, optionally by segment |
| `save_retention_action` | Persists the recommended action to Supabase |
| `get_unactioned_persuadables` | Highest-ROI Persuadables with no action yet |

---

## Tech Stack

### Python Pipeline

| Library | Version | Why |
|---|---|---|
| scikit-learn | ≥1.7.0 | Clustering, calibration, preprocessing |
| catboost | ≥1.2.0 | Per-segment churn classifiers — gradient boosting with built-in categorical handling, chosen over XGBoost for calibration stability in this version |
| xgboost | 3.2.0 | Base learners for the uplift T-Learner and S-Learner (CausalML requirement) |
| causalml | 0.16.0 | T-Learner + S-Learner uplift modeling — Uber's open-source library for this specific problem |
| shap | 0.47.2 | Feature importance approximation (gain-based, not TreeExplainer, due to XGBoost 3.x API instability) |
| pacmap | ≥0.7.0 | 2D behavioral space visualization — faster than UMAP at this scale |
| mlflow | 3.11.1 | Per-segment experiment tracking |
| fastapi | 0.129.0 | Model serving REST API |
| pydantic | 2.12.5 | Request validation for the scoring endpoint |
| groq | ≥0.9.0 | LLM inference (free tier) |
| psycopg2-binary | ≥2.9.9 | PostgreSQL persistence for the audit trail |
| streamlit | ≥1.28.0 | Prototype dashboard (the Next.js dashboard is the production version) |

### Next.js Dashboard

| Library | Version | Why |
|---|---|---|
| next | 16.2.9 | App Router, server components, API routes |
| react | 19.2.4 | UI |
| @supabase/supabase-js | ^2.108.2 | Database client — two instances (anon key for reads, service role key for server-side writes) |
| groq-sdk | ^1.3.0 | AI agent inference |
| recharts | ^3.9.0 | Bar/line/area charts |
| react-plotly.js | ^4.0.0 | Scatter plots (PaCMAP, uplift) |
| tailwindcss | ^4 | Styling |
| typescript | ^5 | Type safety throughout |

---

## Skills Demonstrated

*(Mapped from what the repo actually contains — not aspirational)*

- **Data engineering / ETL pipeline design** — three separate feature engineering modules, schema validation, median imputation, composite feature construction from raw behavioral columns
- **Production ML / MLOps** — FastAPI serving endpoint (`/health`, `/readiness`, `/score`) separate from training code; MLflow experiment tracking; artifact caching with cache-invalidation logic
- **System design & architecture** — documented rationale for every major technical decision (per-segment models, isotonic calibration, observational uplift proxy, server-side writes)
- **LLM application development — agentic systems** — 12-tool ReAct agent with multi-round tool calling, dynamic system prompt from DB configuration, two operating modes (batch and chat)
- **RESTful API design** — FastAPI endpoint with Pydantic validation, liveness/readiness probes, typed request/response schemas
- **Database design** — 5-table Supabase schema, 10 PostgreSQL RPC functions, Row Level Security with separate read (anon) and write (service role) clients
- **Containerization** — Dockerfile with non-root user, layer caching (dependencies before code), Streamlit healthcheck probe
- **CI/CD pipeline** — GitHub Actions: pytest on every push/PR to main, dependency vulnerability audit (`pip-audit`), Docker image build gated on test pass
- **Automated testing** — 3 test modules (feature engineering, churn scoring, FastAPI endpoint), unit tests for boundary conditions on uplift classification thresholds

---

## Getting Started

### Prerequisites

- Python 3.12+
- Node.js 18+
- [Supabase](https://supabase.com) project (free tier)
- [Groq](https://console.groq.com) API key (free tier)
- Kaggle credentials (only for downloading the raw dataset)

### 1. Clone and configure

```bash
git clone https://github.com/shiva-shivanibokka/Churn-Intelligence-Platform.git
cd Churn-Intelligence-Platform

cp .env.example .env
# Edit .env — see Environment Variables section below
```

### 2. Run the ML pipeline

```bash
pip install -r requirements.txt

# Download the default e-commerce dataset
kaggle datasets download \
  -d ankitverma2010/ecommerce-customer-churn-analysis-and-prediction \
  -p data/raw --unzip

# Run the full pipeline (feature engineering → segmentation → churn → uplift)
python src/pipeline.py

# Force full retrain (ignores cached artifacts):
python src/pipeline.py --force

# Run on a different dataset:
python src/pipeline.py --dataset olist      # Brazilian e-commerce, 42K customers
python src/pipeline.py --dataset cell2cell  # Telecom churn, ~71K customers
```

Artifacts are cached to `data/processed/` and `models/`. Subsequent runs without `--force` load from cache in seconds.

### 3. Set up Supabase tables

In your Supabase SQL editor, run `supabase/config_tables.sql` to create the `retention_playbook` and `business_config` tables.

The `customers`, `retention_actions`, and `intervention_feedback` tables should be created to match the schema in `dashboard/src/lib/supabase.ts`.

### 4. Launch the dashboard

```bash
cd dashboard
npm install
npm run dev
# → http://localhost:3000
```

Environment variables are read from the root `.env` file automatically — no `dashboard/.env.local` needed. This is configured via `dashboard/next.config.ts`.

### 5. (Optional) Run the FastAPI scoring endpoint

```bash
uvicorn api.serve:app --host 0.0.0.0 --port 8000

# Liveness check:
curl http://localhost:8000/health
# → {"status": "ok"}

# Score a customer:
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{"Tenure": 12, "HourSpendOnApp": 3.0, "SatisfactionScore": 3, ...}'
# → {"segment": "Champions", "churn_probability": 0.18, "risk_tier": "Low Risk", ...}
```

### 6. (Optional) Docker

```bash
docker build -t churn-engine .
docker run -p 8501:8501 --env-file .env churn-engine
# → Streamlit app at http://localhost:8501
```

---

## Environment Variables

A single `.env` at the repo root is read by both the Python pipeline and the Next.js dashboard.

```bash
# Supabase — all three values from: Project → Settings → API
NEXT_PUBLIC_SUPABASE_URL=https://your-project-id.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=eyJ...      # browser-safe, read-only
SUPABASE_SERVICE_ROLE_KEY=eyJ...          # server-side only, bypasses RLS for writes

# Direct PostgreSQL connection — Project → Settings → Database → URI
DATABASE_URL=postgresql://postgres:password@db.your-project.supabase.co:5432/postgres

# Groq (free tier at console.groq.com)
GROQ_API_KEY=gsk_...

# Kaggle (only needed to download raw datasets)
KAGGLE_USERNAME=your-username
KAGGLE_KEY=your-api-key
```

---

## Usage Examples

**Pipeline output after `python src/pipeline.py`:**

```
[Stage 1] Feature Engineering — 5,630 customers, 8 composite features
[Stage 2] Segmentation — k=5, stability mean ARI=0.921 (Highly Stable)
[Stage 3] Churn Prediction — per-segment CatBoost, AUC 0.945–0.984
[Stage 4] Uplift Modeling — 710 Persuadables, 238 Lost Causes

CustomerType distribution:
  Sure Thing      4,151
  Persuadable       710
  Lost Cause        238
  Sleeping Dog       531
```

**Classify a customer programmatically (from `uplift_model.py`):**

```python
from uplift_model import classify_customer_type

# Thresholds: uplift >= 0.05, churn_prob >= 0.30 → Persuadable
classify_customer_type(uplift_score=0.12, churn_prob=0.65)  # → "Persuadable"
classify_customer_type(uplift_score=-0.08, churn_prob=0.70) # → "Lost Cause"
classify_customer_type(uplift_score=0.10, churn_prob=0.15)  # → "Sure Thing"

# Custom thresholds:
classify_customer_type(uplift_score=0.03, churn_prob=0.50,
                       uplift_threshold=0.02, churn_threshold=0.40)
```

**Score a customer via the FastAPI endpoint:**

```bash
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{
    "Tenure": 12.0, "CityTier": 1, "WarehouseToHome": 15.0,
    "HourSpendOnApp": 3.0, "NumberOfDeviceRegistered": 3,
    "SatisfactionScore": 3, "NumberOfAddress": 2, "Complain": 0,
    "OrderAmountHikeFromlastYear": 15.0, "CouponUsed": 1.0,
    "OrderCount": 3.0, "DaySinceLastOrder": 5.0,
    "CashbackAmount": 150.0, "PreferredLoginDevice": "Mobile Phone",
    "PreferredPaymentMode": "Debit Card", "Gender": "Male",
    "PreferedOrderCat": "Laptop & Accessory", "MaritalStatus": "Single"
  }'
```

```json
{
  "segment": "Loyal Customers",
  "churn_probability": 0.21,
  "churn_prediction": 0,
  "risk_tier": "Low Risk",
  "customer_type": "Sure Thing"
}
```

---

## Project Structure

```
Churn-Intelligence-Platform/
├── src/
│   ├── pipeline.py           # Orchestrator — runs all 4 stages, smart artifact caching
│   ├── features.py           # E-commerce feature engineering + schema validation
│   ├── olist_features.py     # Olist (Brazilian marketplace) feature engineering
│   ├── cell2cell_features.py # Cell2Cell telecom feature engineering
│   ├── segmentation.py       # K-Means++, GMM, PaCMAP, bootstrap ARI stability
│   ├── churn_model.py        # Per-segment CatBoost + isotonic calibration + MLflow
│   ├── uplift_model.py       # T-Learner + S-Learner (CausalML) + ROI ranking
│   ├── retention_llm.py      # Groq-backed retention action generator
│   ├── agent_loop.py         # ReAct agent loop
│   ├── agent_tools.py        # Tool implementations for the agent
│   ├── database.py           # PostgreSQL persistence layer (graceful degradation)
│   └── logging_config.py     # Structured logging configuration
│
├── api/
│   └── serve.py              # FastAPI scoring endpoint (/health, /readiness, /score)
│
├── tests/
│   ├── test_features.py      # Unit tests: schema validation, imputation, feature engineering
│   ├── test_churn_model.py   # Unit tests: churn scoring, customer type classification
│   └── test_api.py           # Integration tests: FastAPI endpoint with mocked models
│
├── dashboard/                # Next.js 16 production dashboard
│   ├── src/
│   │   ├── app/
│   │   │   ├── segmentation/page.tsx
│   │   │   ├── churn/page.tsx
│   │   │   ├── uplift/page.tsx
│   │   │   ├── retention/page.tsx
│   │   │   ├── analytics/page.tsx
│   │   │   └── api/agent/route.ts  # 12-tool ReAct agent
│   │   ├── components/pages/       # Client components (charts, agent UI, audit)
│   │   └── lib/
│   │       ├── data.ts             # Typed Supabase RPC wrappers
│   │       └── supabase.ts         # Client init + TypeScript types
│   └── next.config.ts              # Loads root .env via dotenv
│
├── supabase/
│   └── config_tables.sql     # DDL + seed data for retention_playbook, business_config
│
├── data/processed/           # Pipeline output parquets (tracked in git — no retraining needed to run dashboard)
├── models/                   # Serialized model artifacts (tracked in git)
├── Dockerfile                # python:3.12-slim, non-root user, Streamlit healthcheck
├── .github/workflows/ci.yml  # pytest + pip-audit + Docker build on every push
├── requirements.txt
├── requirements-dev.txt      # Additional test/dev dependencies
├── .env.example
└── README.md
```

---

## Database Schema

### Tables

| Table | Purpose |
|---|---|
| `customers` | Enriched ML output — one row per customer, all features + model scores |
| `retention_actions` | Audit log of every AI-generated recommendation |
| `intervention_feedback` | CSM outcome feedback (`retained` / `churned` / `pending`) |
| `retention_playbook` | DB-driven playbook for `search_retention_playbook` tool — edit rows, no code deploy needed |
| `business_config` | Runtime key-value config: assumed CLV, valid intervention types, channels, timing options |

### Supabase RPC Functions

`get_segment_summary` · `get_churn_kpis(p_segment)` · `get_churn_histogram(p_segment)` · `get_risk_summary` · `get_shap_summary(p_segment)` · `get_avg_churn_by_segment` · `get_customer_type_summary` · `get_roi_by_segment` · `get_top_persuadables(p_limit)` · `get_uplift_kpis`

---

## Testing

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Three test modules, all run automatically via GitHub Actions on every push and pull request to `main`:

| File | What it tests |
|---|---|
| `test_features.py` | Schema validation (missing columns, row count, missing-rate warnings), median imputation, categorical encoding, all 8 engineered features present and in expected ranges |
| `test_churn_model.py` | Four-quadrant customer type classification including boundary conditions on uplift and churn thresholds; `score_customers` column output and risk tier mapping |
| `test_api.py` | FastAPI `/health`, `/readiness`, `/score` endpoints; response field presence, type correctness, 422 on invalid input — using mocked model artifacts |

The CI workflow also runs `pip-audit` for dependency vulnerability scanning (non-blocking, `continue-on-error: true`) and builds the Docker image after tests pass.

No frontend test suite exists for the Next.js dashboard. TypeScript compilation (`tsc --noEmit`) passes with zero errors as of the last push.

---

## Deployment

**Docker (local):**

```bash
docker build -t churn-engine .
docker run -p 8501:8501 --env-file .env churn-engine
```

The Dockerfile uses `python:3.12-slim`, runs as a non-root user, and includes a Streamlit healthcheck. CI builds the image on every push (gated on tests passing) but does not push to a registry or deploy anywhere.

**Not yet deployed to a cloud provider.** The system is designed to be deployable to any Docker-compatible platform (Railway, Render, Fly.io, AWS ECS, GCP Cloud Run). The Next.js dashboard is compatible with Vercel, with one note: environment variables must be set on the platform directly — the root `.env` trick that works locally doesn't apply in cloud deployments.

---

## Results

All numbers below come from running the pipeline on the default e-commerce dataset (5,630 customers). They are reproducible by running `python src/pipeline.py`.

| Metric | Value |
|---|---|
| Cluster stability (bootstrap ARI, 100 resamplings) | 0.921 — "Highly Stable" |
| CV AUC range across 5 segments | 0.945 – 0.984 |
| Persuadables identified (of 948 high-risk customers) | 710 |
| Lost Causes avoided | 238 |

| Segment | Customers | Churn Rate | CV AUC |
|---|---|---|---|
| At-Risk | 1,228 | 23.1% | 0.982 |
| Lapsed | 1,482 | 6.3% | 0.974 |
| Price Sensitive | 889 | 40.4% | 0.945 |
| Champions | 734 | 12.8% | 0.976 |
| Loyal Customers | 1,297 | 9.0% | 0.984 |

---

## Roadmap / Known Limitations

- **Observational uplift, not experimental.** The uplift models use behavioral proxies as treatment indicators (`Complain`, `CouponUsed`) rather than actual A/B test data. This is a documented limitation — production systems train uplift models on randomized experiment logs. The classification thresholds (`uplift ≥ 0.05`, `churn_prob ≥ 0.30`) are tunable via `classify_customer_type()` arguments.
- **No cloud deployment.** The pipeline artifacts (`data/processed/`, `models/`) are committed to git so the dashboard runs without retraining, but the system isn't deployed to a cloud provider. Next step: container push to a registry + cloud run deployment.
- **No frontend test suite.** The Next.js dashboard has no automated tests. `tsc --noEmit` passes, but component behavior is untested.
- **Groq free-tier rate limits.** The AI agent uses Groq's free tier (100,000 tokens/day). Sustained multi-user usage would require a paid tier or model-switching logic.
- **Single-tenant Supabase setup.** Row Level Security is enabled but the current RLS policies are not documented. Multi-tenant use would require policy review.

---

## Industry Parallels

| This Project | Production System |
|---|---|
| Per-segment CatBoost with isotonic calibration | Salesforce Einstein per-tier customer health scoring |
| CausalML T-Learner + S-Learner | Uber's production retention campaign targeting |
| Bootstrap ARI cluster stability (100 resamplings) | Production ML segment validation |
| GMM soft probability assignments | Boundary handling for ambiguous health score tiers |
| 12-tool ReAct AI agent for retention playbooks | Salesforce Einstein Copilot CSM recommendations |
| PaCMAP behavioral space visualization | Netflix member segment exploration |
| DB-driven agent system prompt (`business_config`) | Production LLM config without code deploys |

---

## License

This repository is currently unlicensed. All rights reserved by the author.
