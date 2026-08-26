"""
FastAPI Model Serving Layer
============================
Exposes the trained per-segment churn models as a REST API.

Endpoints:
  GET  /health  — liveness check for deployment platforms
  POST /score   — score a single customer and return churn risk + customer type

This serving layer is the MLOps gap that separates a notebook model from a
production system: the models trained by pipeline.py can now be called from
any downstream service (CRM, marketing automation, data warehouse ETL) without
re-running the full pipeline.

Run locally:
  uvicorn api.serve:app --reload --port 8000

Example request:
  curl -X POST http://localhost:8000/score \
    -H "Content-Type: application/json" \
    -d '{"Tenure": 12, "CityTier": 1, "WarehouseToHome": 15, ...}'
"""

import json
import logging
import os
import sys

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Allow imports from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from features import encode_categoricals, engineer_features, impute_missing
from logging_config import configure_logging
from uplift_model import classify_customer_type

configure_logging()
logger = logging.getLogger(__name__)

MODELS_PATH = os.path.join(os.path.dirname(__file__), "..", "models")

app = FastAPI(
    title="Customer Churn Engine — Scoring API",
    description="Score a customer's churn risk using per-segment CatBoost models.",
    version="2.0.0",
)


# ── Model loading (lazy, cached at first request) ────────────────────────────

_models: dict = {}


def get_models() -> dict:
    """
    Load model artifacts once and cache in memory.

    `label_map.pkl` and `feature_norms.json` are as required as the models
    themselves — without the first this endpoint cannot name the cluster it
    predicts, and without the second it cannot reproduce the features the models
    were trained on. Both used to be missing, and the endpoint answered anyway.
    """
    if not _models:
        required = [
            "segment_models.pkl",
            "kmeans.pkl",
            "scaler.pkl",
            "label_map.pkl",
            "feature_norms.json",
        ]
        for fname in required:
            path = os.path.join(MODELS_PATH, fname)
            if not os.path.exists(path):
                raise RuntimeError(
                    f"Model artifact '{fname}' not found. "
                    "Run `python src/pipeline.py` first to build all models."
                )
        _models["segment_models"] = joblib.load(os.path.join(MODELS_PATH, "segment_models.pkl"))
        _models["kmeans"] = joblib.load(os.path.join(MODELS_PATH, "kmeans.pkl"))
        _models["scaler"] = joblib.load(os.path.join(MODELS_PATH, "scaler.pkl"))
        _models["label_map"] = joblib.load(os.path.join(MODELS_PATH, "label_map.pkl"))
        with open(os.path.join(MODELS_PATH, "feature_norms.json"), encoding="utf-8") as fh:
            _models["norms"] = json.load(fh)

        # Which dataset trained the artifacts currently on disk. This matters to
        # the caller: the request schema below uses e-commerce field names, but
        # the committed artifacts are trained on Cell2Cell, where those names
        # carry proxy meanings (`DaySinceLastOrder` is days on the current
        # handset, `Complain` is >3 care calls, and so on — see
        # src/cell2cell_features.py). Same field, different question.
        meta_path = os.path.join(MODELS_PATH, "pipeline_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, encoding="utf-8") as fh:
                _models["dataset"] = json.load(fh).get("dataset", "unknown")
        else:
            _models["dataset"] = "unknown"

        # Optional: present whenever the uplift stage has been run. Its absence
        # is reported in the response rather than papered over with a zero.
        uplift_path = os.path.join(MODELS_PATH, "uplift_learners.pkl")
        _models["uplift"] = joblib.load(uplift_path) if os.path.exists(uplift_path) else None

        logger.info("Model artifacts loaded from %s", MODELS_PATH)
    return _models


def predict_uplift(models: dict, df: pd.DataFrame) -> float | None:
    """
    Uplift for a single customer, in this project's convention: positive means
    the intervention is expected to REDUCE churn.

    Returns None when the uplift stage has not been run, so the caller can say
    so instead of substituting a number.
    """
    bundle = models.get("uplift")
    if not bundle:
        return None

    feature_cols = bundle["feature_cols"]
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        logger.warning("Cannot score uplift — missing features: %s", missing)
        return None

    X = df[feature_cols].values
    learners = bundle["learners"]

    if "t_learner" in learners:
        # CausalML returns mu_1 - mu_0; this project stores mu_0 - mu_1.
        t = -learners["t_learner"].predict(X).flatten()[0]
        s_ = -learners["s_learner"].predict(X).flatten()[0]
        return float((t + s_) / 2.0)

    p_control = learners["clf_t0"].predict_proba(X)[:, 1][0]
    p_treated = learners["clf_t1"].predict_proba(X)[:, 1][0]
    return float(p_control - p_treated)


# ── Request / Response schemas ───────────────────────────────────────────────

class CustomerFeatures(BaseModel):
    """Raw feature values for a single customer — mirrors the source dataset columns."""
    Tenure: float = Field(..., ge=0, description="Months with the platform")
    CityTier: int = Field(..., ge=1, le=3)
    WarehouseToHome: float = Field(..., ge=0)
    HourSpendOnApp: float = Field(..., ge=0)
    NumberOfDeviceRegistered: int = Field(..., ge=1)
    SatisfactionScore: int = Field(..., ge=1, le=5)
    NumberOfAddress: int = Field(..., ge=1)
    Complain: int = Field(..., ge=0, le=1)
    OrderAmountHikeFromlastYear: float = Field(..., ge=0)
    CouponUsed: float = Field(..., ge=0)
    OrderCount: float = Field(..., ge=0)
    DaySinceLastOrder: float = Field(..., ge=0)
    CashbackAmount: float = Field(..., ge=0)
    PreferredLoginDevice: str = Field(default="Mobile Phone")
    PreferredPaymentMode: str = Field(default="Debit Card")
    Gender: str = Field(default="Male")
    PreferedOrderCat: str = Field(default="Laptop & Accessory")
    MaritalStatus: str = Field(default="Single")

    model_config = {"json_schema_extra": {
        "example": {
            "Tenure": 12, "CityTier": 1, "WarehouseToHome": 15,
            "HourSpendOnApp": 3.0, "NumberOfDeviceRegistered": 3,
            "SatisfactionScore": 3, "NumberOfAddress": 2, "Complain": 0,
            "OrderAmountHikeFromlastYear": 15.0, "CouponUsed": 1.0,
            "OrderCount": 3.0, "DaySinceLastOrder": 5.0, "CashbackAmount": 150.0,
            "PreferredLoginDevice": "Mobile Phone", "PreferredPaymentMode": "Debit Card",
            "Gender": "Male", "PreferedOrderCat": "Laptop & Accessory",
            "MaritalStatus": "Single",
        }
    }}


class ScoreResponse(BaseModel):
    segment: str
    churn_probability: float
    churn_prediction: int
    risk_tier: str
    customer_type: str
    # None when the uplift stage has not been run. Reporting the absence beats
    # substituting 0.0, which is what previously forced customer_type to one of
    # the two negative-uplift quadrants for every request.
    uplift_score: float | None = None
    # The training population behind this answer. The request fields are named
    # for the e-commerce schema; under other datasets they are proxies, so the
    # caller needs to know which one is loaded to read the numbers correctly.
    trained_on: str = "unknown"
    model_version: str = "2.0.0"


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", tags=["ops"])
def health() -> dict:
    """Liveness check — returns 200 if the service is running."""
    return {"status": "ok"}


@app.get("/readiness", tags=["ops"])
def readiness() -> dict:
    """Readiness check — returns 200 only if model artifacts are loaded."""
    try:
        models = get_models()
        return {
            "status": "ready",
            "models_loaded": True,
            "trained_on": models["dataset"],
            "segments": sorted(models["label_map"].values()),
            "uplift_available": models["uplift"] is not None,
        }
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/score", response_model=ScoreResponse, tags=["inference"])
def score(customer: CustomerFeatures) -> ScoreResponse:
    """
    Score a single customer and return their churn risk profile.

    The customer is assigned to a segment with the trained K-Means model, scored
    by that segment's own CatBoost classifier, and given an uplift estimate from
    the fitted T/S-learners — so `segment` and `customer_type` are predictions,
    not constants.

    They were constants. This endpoint used to compute the K-Means cluster and
    then discard it, taking `list(segment_models)[0]` instead, so every customer
    came back "Champions"; and it passed a hardcoded `uplift_score=0.0` into
    `classify_customer_type`, which is below the 0.05 threshold, so
    `customer_type` could only ever be "Lost Cause" or "Sleeping Dog". The two
    values the README documented as example output were both unreachable.
    """
    models = get_models()
    segment_models = models["segment_models"]
    kmeans = models["kmeans"]
    scaler = models["scaler"]
    label_map = models["label_map"]

    # Build a one-row DataFrame matching the pipeline's column format.
    df = pd.DataFrame([customer.model_dump()])

    # Same preprocessing as training — and crucially the *same* normalising
    # constants, loaded rather than refitted. Refitting them on one row divides
    # each feature by itself.
    df = impute_missing(df)
    df = encode_categoricals(df)
    df = engineer_features(df, norms=models["norms"])

    # ── Segment assignment ───────────────────────────────────────────────────
    # Prefer the names the scaler was fitted with, so the order always matches
    # training, and fall back to the canonical list when it was fitted on a bare
    # numpy array and has no names.
    #
    # Tested against `is not None` rather than truthiness: sklearn stores this as
    # a numpy array, and `array or default` raises "truth value of an array with
    # more than one element is ambiguous" — on every real request, while a mock
    # that returns None sails through.
    fitted_names = getattr(scaler, "feature_names_in_", None)
    clustering_features = list(fitted_names) if fitted_names is not None else [
        "EngagementScore", "RecencySignal", "StickinessIndex", "SpendTrend",
        "SupportRiskScore", "DiscountSensitivity", "TenureStability", "WarehouseFriction",
        "CityTier", "HourSpendOnApp", "OrderCount", "NumberOfDeviceRegistered",
        "SatisfactionScore",
    ]
    missing = [f for f in clustering_features if f not in df.columns]
    if missing:
        raise HTTPException(
            status_code=500,
            detail=f"Cannot build clustering features {missing} from this request.",
        )

    X_scaled = scaler.transform(df[clustering_features])
    raw_cluster = int(kmeans.predict(X_scaled)[0])

    segment_name = label_map.get(raw_cluster)
    if segment_name is None:
        raise HTTPException(
            status_code=500,
            detail=f"K-Means returned cluster {raw_cluster}, which has no segment name "
                   f"in label_map.pkl (known: {sorted(label_map)}).",
        )

    model_dict = segment_models.get(segment_name)
    if model_dict is None:
        raise HTTPException(
            status_code=500,
            detail=f"No trained model for segment '{segment_name}' — it was skipped "
                   "at training time for having too few rows or a single class.",
        )

    # ── Churn probability from that segment's model ──────────────────────────
    # Take the feature list off the model itself, in its trained order. A
    # hardcoded list here silently disagrees with any dataset but the one it was
    # written for, and CatBoost is happy to score a mis-ordered frame.
    churn_feature_cols = model_dict["feature_cols"]
    missing = [f for f in churn_feature_cols if f not in df.columns]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Request cannot produce features the '{segment_name}' model was "
                   f"trained on: {missing}.",
        )

    df["Segment"] = segment_name
    proba = float(model_dict["calibrated_clf"].predict_proba(df[churn_feature_cols])[:, 1][0])
    prediction = int(proba >= 0.5)

    if proba < 0.3:
        risk_tier = "Low Risk"
    elif proba < 0.6:
        risk_tier = "Medium Risk"
    else:
        risk_tier = "High Risk"

    # ── Uplift and the four-quadrant type ────────────────────────────────────
    uplift = predict_uplift(models, df)
    if uplift is None:
        customer_type = "Unknown — uplift models not available"
    else:
        customer_type = classify_customer_type(uplift_score=uplift, churn_prob=proba)

    logger.info(
        "Scored customer — cluster=%d segment=%s churn_prob=%.3f uplift=%s risk=%s",
        raw_cluster, segment_name, proba,
        "n/a" if uplift is None else f"{uplift:.4f}", risk_tier,
    )

    return ScoreResponse(
        segment=segment_name,
        churn_probability=round(proba, 4),
        churn_prediction=prediction,
        risk_tier=risk_tier,
        customer_type=customer_type,
        uplift_score=None if uplift is None else round(uplift, 4),
        trained_on=models["dataset"],
    )
