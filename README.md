# Customer Segmentation & Churn Engine

> End-to-end decision intelligence platform: behavioral segmentation → per-cohort churn prediction → uplift modeling → 12-tool AI retention agent → closed-loop outcome tracking.

![CI](https://github.com/shiva-shivanibokka/Churn-Intelligence-Platform/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.12-blue)
![Docker](https://img.shields.io/badge/docker-build-passing)
![Deployed](https://img.shields.io/badge/dashboard-live%20on%20Vercel-black)

**Live dashboard:** [customer-segmentation-churn.vercel.app](https://customer-segmentation-churn.vercel.app/retention)

---

## Recruiter TL;DR

- **What it does:** Segments customers behaviorally, predicts churn per cohort with calibrated probabilities, identifies the subset worth spending retention budget on (uplift modeling), and deploys a 12-tool ReAct AI agent that reasons over SHAP drivers, intervention history, and ROI before generating and saving a personalized retention plan.
- **Hardest problem solved:** Replacing a naive "email everyone above 0.7 churn probability" approach with causal uplift modeling (CausalML T-Learner + S-Learner) to distinguish Persuadables from Lost Causes and Sleeping Dogs — the same targeting logic Uber open-sourced CausalML to solve.
- **Verified results, generated from the artifacts rather than typed:** holdout AUC 0.576–0.692 across 5 segments; isotonic calibration cutting held-out Brier from 0.2342 to 0.1946; cluster stability mean ARI 0.915 over 100 seeded bootstrap resamplings; 7,602 Persuadables on the Cell2Cell dataset. Every one of those figures is written by `scripts/readme_metrics.py` and re-checked in CI — see [Results](#results).
- **The bug worth reading about:** the uplift score's sign was inverted, so for months the "Persuadables" this system recommended contacting were precisely the customers its own model predicted contact would drive away. It produced a ranked list, plausible ROI and a working dashboard the entire time. [What happened, and the check that now runs every pipeline](#results).

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
  → Per-segment CatBoost classifiers, early-stopped + isotonic calibration
  → Exact CatBoost TreeSHAP, per customer, signed
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
- **Isotonic calibration over raw probabilities.** These models are trained with `class_weights=[1, pos_weight]` to handle imbalance, which inflates the positive class *by construction* — so a raw score of 0.7 is not a 70% chance of churn. That matters because the score is then multiplied by CLV to rank retention spend, which is exactly the case where an uncalibrated probability costs money. Isotonic is fitted on a held-out slice and preferred over Platt scaling for non-parametric distributions.

  The claim used to be false. `calibrated_clf` was a plain alias for the base model, on the reasoning that CatBoost is well calibrated natively — which ignores the class weighting, and which nothing checked because every call site read `model_dict["calibrated_clf"]` and so looked calibrated either way. `holdout_brier_uncalibrated` and `holdout_brier` are now both recorded, so it is a measurement.
- **Observational uplift instead of A/B targeting.** The datasets don't include historical experiment logs. Treatment proxies (`Complain` flag = received support outreach; `CouponUsed > 0` = received discount) follow the academic literature on observational uplift. Production systems (Uber, Netflix) train on actual randomized experiment logs.
- **Server-side retention action saves.** The `retention_actions` table has Row Level Security enabled in Supabase. The AI agent API route uses the service role key (server-side only) for inserts — the browser anon key is read-only.
- **DB-driven agent configuration.** The system prompt is rebuilt from the `business_config` table on every request. Changing CLV assumptions, intervention types, or outreach channels requires only a database row update — no code change, no redeploy.

---

## Features

**ML Pipeline**
- 8 engineered composite features: `EngagementScore`, `RecencySignal`, `StickinessIndex`, `SpendTrend`, `SupportRiskScore`, `DiscountSensitivity`, `TenureStability`, `WarehouseFriction`
- Schema validation with column presence and missing-rate checks before any transformation
- Supports 3 datasets selectable via `--dataset` CLI flag: e-commerce (5,630 customers), Olist Brazilian marketplace (42,325), Cell2Cell telecom (51,047). Cached artifacts record which dataset built them, so switching datasets rebuilds rather than silently returning the previous one
- K-Means++ with 5 clusters + GMM soft probability assignments (each customer gets a probability distribution across segments, not just a hard label)
- Bootstrap cluster stability: Adjusted Rand Index across 100 resamplings with a pass/warn/fail grading scheme
- Per-segment CatBoost classifiers with a stratified 80/20 holdout, 3-fold CV, and early stopping against the calibration slice
- Isotonic probability calibration fitted on a held-out slice, with the before/after Brier recorded per segment
- MLflow experiment tracking — one run logged per segment per training
- CausalML T-Learner + S-Learner uplift models with four-quadrant customer classification
- Intervention ROI ranking: `net_roi = uplift_score × CLV − intervention_cost`

**FastAPI Scoring Endpoint**
- `GET /health` — liveness probe
- `GET /readiness` — readiness probe (confirms model artifacts loaded)
- `POST /score` — accepts raw customer features, returns `segment` (from the trained K-Means model), `churn_probability` (calibrated, from that segment's own classifier), `risk_tier`, `customer_type` (from the fitted uplift learners) and `trained_on`
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
python src/pipeline.py --dataset cell2cell  # Telecom churn, 51,047 customers
```

Artifacts are cached to `data/processed/` and `models/`. Subsequent runs without `--force` load from cache in seconds.

### 3. Set up Supabase tables

With `DATABASE_URL` set in `.env`, one command builds the whole backend:

```bash
python restore_supabase.py
```

It is idempotent, so it is safe to re-run, and it verifies rather than assumes:
it counts the rows, calls all ten RPCs, and exits non-zero if any of them fail
or come back empty.

Order matters, which is the reason this script exists. The RPCs and the RLS
policies both reference `customers`, so they have to run *after* the migration
that creates it. The script runs:

1. `src/database.py::_create_schema` — `conversations`, `messages`, `retention_actions`, `intervention_feedback`.
2. `supabase/config_tables.sql` — `retention_playbook` and `business_config`.
3. `migrate_to_supabase.py` — creates `customers` and loads it from `data/processed/uplift.parquet`.
4. `supabase/rpc_functions.sql` — the 10 `SECURITY DEFINER` aggregation functions the dashboard calls.
5. `supabase/rls_policies.sql` — **enables Row-Level Security** on all five tables and adds the access policies (anon read-only + feedback insert; service-role bypasses RLS for server-side writes). This is required: the dashboard ships the public anon key in the browser, so without RLS that key would grant anyone full read/write/delete on your data.

To run any step by hand instead, paste the `supabase/*.sql` files into the
Supabase SQL editor in the order above.

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

# Readiness — reports what it is ready to serve, not just 200:
curl http://localhost:8000/readiness
# → {"status":"ready","trained_on":"cell2cell","segments":[...],"uplift_available":true}

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

# Shared secret for the daily keepalive cron (see Deployment)
CRON_SECRET=any-long-random-string
```

---

## Usage Examples

**Pipeline output after `python src/pipeline.py --dataset cell2cell`:**

```
[Stage 1] Feature Engineering — 51,047 customers, 8 composite features
[Stage 2] Segmentation — k=5, stability mean ARI=0.915 (Highly Stable)
[Stage 3] Churn Prediction — per-segment CatBoost, holdout AUC 0.576–0.692
[Stage 4] Uplift Modeling — 7,602 Persuadables, 18,222 Lost Causes
          Uplift direction check passed — treated top decile churns 0.255 vs bottom 0.425

CustomerType distribution:
  Sleeping Dog      22,492
  Lost Cause        18,222
  Persuadable        7,602
  Sure Thing         2,731
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
  "segment": "At-Risk",
  "churn_probability": 0.3176,
  "churn_prediction": 0,
  "risk_tier": "Medium Risk",
  "customer_type": "Lost Cause",
  "uplift_score": -0.1045,
  "trained_on": "cell2cell",
  "model_version": "2.0.0"
}
```

That is copied from an actual call against the committed artifacts, which is
worth saying because the block that used to sit here was not. It showed
`"segment": "Loyal Customers"` and `"customer_type": "Sure Thing"`, and the
endpoint could produce neither: it computed the K-Means cluster and then
discarded it in favour of `list(segment_models)[0]`, so every customer came back
"Champions", and it passed a hardcoded `uplift_score=0.0` into the classifier,
which sits below the Persuadable threshold and left only the two negative-uplift
quadrants reachable. Both are fixed, and `tests/test_api.py` now asserts that
each cluster maps to its own segment and that all four quadrants are reachable —
the previous fixture had exactly one segment in it, which is why nothing caught
either.

**`trained_on` matters when reading the numbers.** The request fields are named
for the e-commerce schema, but the committed artifacts are trained on Cell2Cell,
where those names carry proxy meanings — `DaySinceLastOrder` is days on the
current handset, `Complain` is more than three care calls, `SatisfactionScore`
is a credit rating. The mapping is in `src/cell2cell_features.py`.

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
│   │   │   ├── page.tsx                # Segmentation (root route /)
│   │   │   ├── churn/page.tsx
│   │   │   ├── uplift/page.tsx
│   │   │   ├── retention/page.tsx
│   │   │   ├── analytics/page.tsx
│   │   │   ├── error.tsx               # Shown when a data load throws (see Deployment)
│   │   │   ├── api/agent/route.ts      # 12-tool ReAct agent (Vercel serverless)
│   │   │   └── api/keepalive/route.ts  # Daily cron — stops Supabase pausing the project
│   │   ├── components/pages/       # Client components (charts, agent UI, audit)
│   │   └── lib/
│   │       ├── data.ts             # Typed Supabase RPC wrappers
│   │       └── supabase.ts         # Client init + TypeScript types
│   └── next.config.ts              # Loads root .env via dotenv
│
├── supabase/
│   ├── config_tables.sql     # DDL + seed data for retention_playbook, business_config
│   ├── rpc_functions.sql     # The 10 SECURITY DEFINER functions the dashboard calls
│   └── rls_policies.sql      # Row-Level Security on all five tables
│
├── restore_supabase.py       # Rebuilds the whole backend in dependency order, then verifies
├── migrate_to_supabase.py    # Creates `customers` and loads it from uplift.parquet
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

**The full system is deployed.** The Next.js dashboard is live on Vercel at [customer-segmentation-churn.vercel.app](https://customer-segmentation-churn.vercel.app/retention). The 12-tool AI agent runs as a Vercel serverless function (Next.js API route, 60-second timeout configured in `dashboard/vercel.json`). All data is served from Supabase. No separate backend deployment is needed — the dashboard is self-contained.

Environment variables (Supabase keys, Groq API key, `CRON_SECRET`) are set directly on the Vercel project — the root `.env` trick that works locally doesn't apply in cloud deployments.

**`SUPABASE_SERVICE_ROLE_KEY` is required, not optional.** RLS blocks inserts made
with the anon key, so without it the agent still generates plans but every write
to `retention_actions` is refused and Audit & Analytics silently stops gaining
rows. The route now reports that back to the browser instead of logging it and
returning 200, but the variable still has to be set.

### Keeping the database awake

Supabase pauses free-tier projects after about a week without traffic, and a
paused project keeps drifting toward deletion at the 90-day mark. A portfolio
demo that gets opened once a month never clears that bar on its own.

`dashboard/src/app/api/keepalive/route.ts` runs one cheap `SELECT` against
`business_config`, and `dashboard/vercel.json` schedules it daily at 06:00 UTC.
Any API request counts as activity, so the project stays permanently inside its
7-day window and the deletion timer never starts.

Two deliberate choices:

- **It fails closed.** With `CRON_SECRET` unset the endpoint rejects everything, including Vercel's own scheduled call. Set the variable on the Vercel project (Production) and confirm the first run under **Vercel → Logs** — an unverified keepalive is indistinguishable from no keepalive.
- **It returns 500 when the database is unreachable**, so a broken keepalive surfaces as a failed cron. An endpoint that returns `200 OK` while the database is gone reports success at exactly the moment it has stopped working.

Cron jobs only run on production deployments, never on previews.

### What happened in August 2026

This section is here because the failure was instructive.

The Supabase project was paused for inactivity. From outside, a paused project
is indistinguishable from a deleted one: DNS for `<ref>.supabase.co` stops
resolving entirely (`NXDOMAIN`, not a timeout), and the connection pooler
answers `FATAL: (ENOTFOUND) tenant/user postgres.<ref> not found`. Both readings
suggest the data is gone. It was not — resuming the project from the Supabase
dashboard brought back all 51,047 rows untouched.

The dashboard made this much worse than it needed to be. Every page wrapped its
queries in `.catch(() => [])`, which collapsed *"the database is unreachable"*
and *"the query returned no rows"* into the same value. So the site did not fail
— it rendered a confident zero-state and announced **"~0 customers grouped into
5 behavioural segments"**, with the `5` hardcoded beside a computed `0`. It
demoed as an empty product rather than a broken one, which is the worse of the
two.

Those catches are gone. Queries now throw to `dashboard/src/app/error.tsx`,
which says the data source is unreachable and shows the error digest — in
production, Next.js withholds a Server Component's real message from the
browser, so the digest is the only thread back to the server logs. Both figures
in the segmentation blurb are now counted from the rows actually returned.

**FastAPI scoring endpoint** (`/health`, `/readiness`, `/score`) is a standalone optional tool for scoring new customers programmatically outside the dashboard. It runs locally or via Docker and is not a dependency for the deployed system.

---

## Results

<!-- BEGIN GENERATED RESULTS -->
<!-- Generated by scripts/readme_metrics.py — do not edit by hand.
     Run `python scripts/readme_metrics.py --write` after any pipeline run. -->

All figures below are read straight out of the committed artifacts by
`scripts/readme_metrics.py`, which CI re-runs in `--check` mode. If a model
changes and this section is not regenerated, the build fails. Dataset:
**cell2cell**, 51,047 customers.

### Per-segment churn models

| Segment | Customers | Churn rate | Holdout AUC | Train AUC | Holdout Brier (raw → calibrated) | Trees |
|---|---|---|---|---|---|---|
| At-Risk | 8,400 | 26.4% | **0.692** | 0.726 | 0.2197 → 0.1778 | 64 |
| Price Sensitive | 12,966 | 30.0% | **0.645** | 0.691 | 0.2327 → 0.2001 | 39 |
| Loyal Customers | 9,553 | 28.0% | **0.614** | 0.672 | 0.2368 → 0.1955 | 26 |
| Champions | 8,153 | 23.9% | **0.591** | 0.759 | 0.2367 → 0.1798 | 95 |
| Lapsed | 11,975 | 33.3% | **0.576** | 0.659 | 0.2453 → 0.2197 | 77 |

**Mean holdout AUC 0.624**, against 0.702 on the rows the
models were fitted on. That gap is the honest one, and reporting the holdout
number is the whole point of the split — an earlier version of this README
quoted 0.789–0.859, which were neither: they were carried over from a run on a
different dataset, next to customer counts that had been updated.

An AUC in the low 0.6s is a modest model, and it is what this data supports.
Cell2Cell churn is famously hard — the published benchmark literature sits in
roughly the same range — and the per-segment split makes it harder still by
giving each model a fifth of the rows and a narrower slice of variation.

### What calibration is worth

**Brier 0.2342 → 0.1946** on held-out rows, a
17% reduction. Calibration cannot change
AUC — it is a monotone map, so the ranking is identical by construction — which
is exactly why the pair of Brier scores is the number that means anything.

The clearest way to see it: predicted churn now averages
**0.2882** against an actual churn rate of
**0.2882**. Uncalibrated, these models are trained with
`class_weights=[1, pos_weight]`, which inflates the positive class on purpose,
and their probabilities are then multiplied by CLV to rank retention spend.

This is also why only **322 customers** are High Risk
(calibrated P(churn) ≥ 0.6) rather than the tens of thousands an earlier version
of this README reported. That larger figure was not a finding; it was the
weighting artifact, read as risk.

| Risk tier | Customers |
|---|---|
| High Risk (≥ 0.60) | 322 |
| Medium Risk (0.30–0.60) | 25,502 |
| Low Risk (< 0.30) | 25,223 |

### Segmentation stability

Bootstrap Adjusted Rand Index over 100 resamplings:
**mean ARI 0.915 ± 0.136** —
*Highly Stable*. The resampling draws from a seeded generator, so this
figure is reproducible; it previously used NumPy's global RNG and moved on every
run while the README quoted it to three decimals.

### Uplift and the four quadrants

| Customer type | Count | Meaning |
|---|---|---|
| Persuadable | 7,602 | High churn risk **and** responds to intervention — the target list |
| Sure Thing | 2,731 | Would respond, but is not at risk — no spend needed |
| Lost Cause | 18,222 | At risk, but intervention does not help |
| Sleeping Dog | 22,492 | Not at risk, and contact makes things worse — do not disturb |

`UpliftScore` is `mu_0 - mu_1`: **positive means the intervention reduces this
customer's churn probability.** That convention is checked against observed
outcomes on every run rather than assumed. Among customers who were actually
treated, the top uplift decile churns at
**0.2554** against the
bottom decile's **0.4246**
— the right way round, and the pipeline fails loudly if it ever is not.

It was not the right way round. CausalML's meta-learners return `mu_1 - mu_0`,
the treatment effect on the *outcome*, and the raw output was used unnegated.
The outcome here is churn, so a large positive score marked the customers
contact was expected to drive away — and `classify_customer_type` read it as
"responds well". Every name on the Persuadable list was the model's strongest
Sleeping Dog, and the AI agent wrote retention plans for all of them. The sign
error produced a ranked list, sensible-looking ROI figures and a full dashboard,
which is why it survived: nothing about it looked wrong.

**Treatment is observational, not experimental.** `Complain` and `CouponUsed`
are proxies for having received outreach, so these are associations under an
assumption of no unmeasured confounding, not causal estimates. Production uplift
models train on randomised experiment logs. The direction check above shares
every confound of the proxy — it catches a flipped sign, which is what it is
for, and it is not evidence of an effect size.
<!-- END GENERATED RESULTS -->

---

## Roadmap / Known Limitations

- **Observational uplift, not experimental.** The uplift models use behavioral proxies as treatment indicators (`Complain`, `CouponUsed`) rather than actual A/B test data. This is a documented limitation — production systems train uplift models on randomized experiment logs. The classification thresholds (`uplift ≥ 0.05`, `churn_prob ≥ 0.30`) are tunable via `classify_customer_type()` arguments.
- **FastAPI scoring endpoint not cloud-deployed.** The dashboard and AI agent are fully deployed (Vercel + Supabase). The FastAPI `/score` endpoint is a standalone tool for programmatic scoring outside the dashboard — it runs locally or via Docker but is not required for the deployed system.
- **No frontend test suite.** The Next.js dashboard has no automated tests. CI runs `tsc --noEmit` and ESLint on every push, so type errors and lint regressions are caught, but component behaviour is untested.
- **Modest AUC, honestly reported.** Holdout AUC sits in the low 0.6s. Cell2Cell is a hard churn dataset and splitting it five ways makes it harder; the models are useful for ranking, not for confident individual predictions. The calibration is what makes the ranking safe to spend money against.
- **The agent endpoint is public and rate-limited in memory.** `/api/agent` takes no auth so anyone opening the demo can use it, which also means anyone can spend the Groq free tier. A per-IP bucket in module scope blunts casual abuse but does not survive across serverless instances; a shared store (Vercel KV, Upstash) is the real fix if this ever mattered.
- **Groq free-tier rate limits.** The AI agent uses Groq's free tier (100,000 tokens/day). Sustained multi-user usage would require a paid tier or model-switching logic.
- **Single-tenant Supabase setup.** Row Level Security is enabled and the policies are version-controlled in `supabase/rls_policies.sql` (anon read-only + feedback insert; service-role for server-side writes). The current policies treat all data as a single shared tenant — multi-tenant use would require per-tenant scoping in the policy predicates.

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

---

Built by Shivani Bokka
