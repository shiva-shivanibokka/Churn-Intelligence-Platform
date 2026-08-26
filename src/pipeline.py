"""
Pipeline Orchestrator
======================
Runs the full 5-stage decision intelligence pipeline end-to-end
and caches all artifacts to disk so the Streamlit UI loads instantly.

Run this once to build all models. The Streamlit app reads from cached
.parquet and .pkl files — no retraining on app load.
"""

import json
import logging
import os
import sys

import joblib
import numpy as np
import pandas as pd

# Add src directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from cell2cell_features import build_cell2cell_pipeline, get_cell2cell_feature_sets
from churn_model import run_churn_pipeline
from composite_features import (
    COMPOSITE_FEATURES,
    add_composite_features,
    fit_composite_norms,
)
from features import build_pipeline, get_feature_sets
from logging_config import configure_logging
from olist_features import build_olist_pipeline, get_olist_feature_sets
from segmentation import run_segmentation
from uplift_model import run_uplift_pipeline

logger = logging.getLogger(__name__)

PROCESSED_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODELS_PATH = os.path.join(os.path.dirname(__file__), "..", "models")

# Which builder and which feature sets belong to each --dataset choice.
DATASETS = {
    "ecommerce": (build_pipeline, get_feature_sets),
    "olist": (build_olist_pipeline, get_olist_feature_sets),
    "cell2cell": (build_cell2cell_pipeline, get_cell2cell_feature_sets),
}

# Every dataset writes to the same artifact paths, so the only way to know which
# one produced the files on disk is to record it next to them.
META_PATH = os.path.join(MODELS_PATH, "pipeline_meta.json")

# Constants the composite features are normalised by, needed by anything that
# scores rows outside the training run (api/serve.py).
NORMS_PATH = os.path.join(MODELS_PATH, "feature_norms.json")


def _cached_dataset() -> str | None:
    """The dataset the artifacts on disk were built from, if it was recorded."""
    try:
        with open(META_PATH, encoding="utf-8") as fh:
            return json.load(fh).get("dataset")
    except (OSError, ValueError):
        return None


def run_full_pipeline(force_retrain: bool = False, dataset: str = "ecommerce") -> dict:
    """
    Run all 5 stages of the decision intelligence pipeline.

    If cached artifacts exist and force_retrain=False, loads from disk.
    Set force_retrain=True to rebuild everything from scratch.
    """
    os.makedirs(PROCESSED_PATH, exist_ok=True)
    os.makedirs(MODELS_PATH, exist_ok=True)

    uplift_path = os.path.join(PROCESSED_PATH, "uplift.parquet")
    seg_profiles_path = os.path.join(MODELS_PATH, "segment_profiles.pkl")
    seg_models_path = os.path.join(MODELS_PATH, "segment_models.pkl")
    stability_path = os.path.join(MODELS_PATH, "stability.pkl")

    segmented_path = os.path.join(PROCESSED_PATH, "segmented.parquet")

    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset {dataset!r}. Choose one of {sorted(DATASETS)}.")
    build_fn, feature_set_fn = DATASETS[dataset]

    # The cache is only a cache for the dataset that filled it.
    #
    # This used to load uplift.parquet whenever the file existed, whatever
    # --dataset asked for, and then return the *e-commerce* feature sets
    # regardless. So `--dataset olist` on a machine holding cell2cell artifacts
    # printed a full, plausible summary of cell2cell — the flag the README
    # documents for switching datasets silently did nothing.
    cached_dataset = _cached_dataset()
    cache_usable = os.path.exists(uplift_path) and cached_dataset == dataset
    if os.path.exists(uplift_path) and cached_dataset != dataset:
        logger.info(
            "Cached artifacts were built from dataset=%s but dataset=%s was "
            "requested — rebuilding rather than returning the wrong data.",
            cached_dataset or "unrecorded", dataset,
        )

    if not force_retrain and cache_usable:
        logger.info("Loading cached artifacts...")
        df = pd.read_parquet(uplift_path)
        seg_profiles = joblib.load(seg_profiles_path)
        segment_models = joblib.load(seg_models_path)
        stability = joblib.load(stability_path)
        feature_sets = feature_set_fn()
        logger.info("Loaded from cache.")
        return {
            "df": df,
            "seg_profiles": seg_profiles,
            "segment_models": segment_models,
            "stability": stability,
            "feature_sets": feature_sets,
        }

    logger.info("=" * 60)
    logger.info("SUBSCRIPTION CHURN ENGINE — FULL PIPELINE [dataset=%s]", dataset)
    logger.info("=" * 60)

    # Stage 1: Feature Engineering
    logger.info("[Stage 1] Feature Engineering")
    feature_sets = feature_set_fn()
    df = build_fn(save=True)

    # ── Persist the composite normalisers ────────────────────────────────────
    #
    # Five of the eight composites divide by a column maximum. During training
    # that maximum is a property of the training population; at inference it has
    # to be the *same* constant, or scoring one customer divides them by
    # themselves. Saving them here is what makes api/serve.py able to reproduce
    # training-time features.
    #
    # The constants are re-derived from the built frame rather than threaded out
    # of the builder, so the assumption that doing so reproduces what the builder
    # used is checked rather than trusted.
    norms = fit_composite_norms(df)
    _replay = add_composite_features(df.copy(), norms)
    for feat in COMPOSITE_FEATURES:
        if feat in df.columns and not np.allclose(
            _replay[feat], df[feat], rtol=1e-9, atol=1e-12, equal_nan=True
        ):
            raise RuntimeError(
                f"Refitting composite norms on the built frame did not reproduce "
                f"'{feat}'. The dataset builder must be transforming a column after "
                f"the composites are computed, so the saved norms would not match "
                f"what the models were trained on."
            )
    with open(NORMS_PATH, "w", encoding="utf-8") as fh:
        json.dump(norms, fh, indent=2)
    logger.info("Saved composite feature norms to %s", NORMS_PATH)

    # Stage 2: Segmentation — skip if cached segmented.parquet exists
    if (
        not force_retrain
        and cached_dataset == dataset
        and os.path.exists(segmented_path)
        and os.path.exists(stability_path)
    ):
        logger.info("[Stage 2] Customer Segmentation — loading from cache")
        df = pd.read_parquet(segmented_path)
        stability = joblib.load(stability_path)
        seg_profiles = (
            df.groupby("Segment")[
                [
                    "EngagementScore",
                    "RecencySignal",
                    "StickinessIndex",
                    "SpendTrend",
                    "SupportRiskScore",
                    "Churn",
                ]
            ]
            .mean()
            .round(3)
        )
        seg_profiles["CustomerCount"] = df.groupby("Segment").size()
        seg_profiles["ChurnRate"] = df.groupby("Segment")["Churn"].mean().round(3)
    else:
        logger.info("[Stage 2] Customer Segmentation")
        seg_results = run_segmentation(df, feature_sets["clustering"], n_clusters=5)
        df = seg_results["df"]
        stability = seg_results["stability"]
        seg_profiles = seg_results["profiles"]
        joblib.dump(seg_profiles, seg_profiles_path)
        joblib.dump(stability, stability_path)

    # Stage 3: Per-Segment Churn Models
    logger.info("[Stage 3] Per-Segment Churn Prediction")
    churn_results = run_churn_pipeline(df, feature_sets["churn_model"])
    df = churn_results["df"]
    segment_models = churn_results["segment_models"]

    # Stage 4: Uplift Modeling
    logger.info("[Stage 4] Uplift Modeling (Causal ML)")
    uplift_results = run_uplift_pipeline(df, feature_sets["uplift_model"])
    df = uplift_results["df"]

    # Save final enriched dataset
    df.to_parquet(uplift_path, index=False)
    with open(META_PATH, "w", encoding="utf-8") as fh:
        json.dump({"dataset": dataset, "n_rows": int(len(df))}, fh, indent=2)
    logger.info("Pipeline complete. Final dataset: %s", df.shape)
    logger.info("Columns: %s", df.columns.tolist())

    return {
        "df": df,
        "seg_profiles": seg_profiles,
        "segment_models": segment_models,
        "stability": stability,
        "feature_sets": feature_sets,
    }


if __name__ == "__main__":
    import argparse

    configure_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Force retrain all models")
    parser.add_argument("--dataset", choices=["ecommerce", "olist", "cell2cell"], default="ecommerce",
                        help="Dataset: 'ecommerce' (original), 'olist' (Brazilian e-commerce), 'cell2cell' (71K subscription churn)")
    args = parser.parse_args()

    results = run_full_pipeline(force_retrain=args.force, dataset=args.dataset)
    df = results["df"]

    logger.info("=" * 60)
    logger.info("PIPELINE SUMMARY")
    logger.info("=" * 60)
    logger.info("Total customers: %d", len(df))
    logger.info("Segment distribution:\n%s", df["Segment"].value_counts().to_string())
    logger.info("Customer type distribution:\n%s", df["CustomerType"].value_counts().to_string())
    logger.info("Risk tier distribution:\n%s", df["RiskTier"].value_counts().to_string())
    logger.info(
        "Persuadables with positive ROI: %d",
        ((df["CustomerType"] == "Persuadable") & (df["NetROI"] > 0)).sum(),
    )
    stab = results["stability"]
    logger.info("Cluster stability: Mean ARI=%.3f ± %.3f | Grade: %s", stab["mean_ari"], stab["std_ari"], stab["grade"])
