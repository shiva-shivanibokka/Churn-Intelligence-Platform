"""
Pipeline Orchestrator
======================
Runs the full 5-stage decision intelligence pipeline end-to-end
and caches all artifacts to disk so the Streamlit UI loads instantly.

Run this once to build all models. The Streamlit app reads from cached
.parquet and .pkl files — no retraining on app load.
"""

import logging
import os
import sys
import joblib
import pandas as pd

# Add src directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from logging_config import configure_logging
from features import build_pipeline, get_feature_sets
from olist_features import build_olist_pipeline, get_olist_feature_sets
from segmentation import run_segmentation
from churn_model import run_churn_pipeline
from uplift_model import run_uplift_pipeline

logger = logging.getLogger(__name__)

PROCESSED_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODELS_PATH = os.path.join(os.path.dirname(__file__), "..", "models")


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

    if not force_retrain and os.path.exists(uplift_path):
        logger.info("Loading cached artifacts...")
        df = pd.read_parquet(uplift_path)
        seg_profiles = joblib.load(seg_profiles_path)
        segment_models = joblib.load(seg_models_path)
        stability = joblib.load(stability_path)
        feature_sets = get_feature_sets()
        logger.info("Loaded from cache.")
        return {
            "df": df,
            "seg_profiles": seg_profiles,
            "segment_models": segment_models,
            "stability": stability,
            "feature_sets": feature_sets,
        }

    logger.info("=" * 60)
    logger.info("CUSTOMER SEGMENTATION & CHURN ENGINE — FULL PIPELINE [dataset=%s]", dataset)
    logger.info("=" * 60)

    # Stage 1: Feature Engineering
    logger.info("[Stage 1] Feature Engineering")
    if dataset == "olist":
        feature_sets = get_olist_feature_sets()
        df = build_olist_pipeline(save=True)
    else:
        feature_sets = get_feature_sets()
        df = build_pipeline(save=True)

    # Stage 2: Segmentation — skip if cached segmented.parquet exists
    if (
        not force_retrain
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
    parser.add_argument("--dataset", choices=["ecommerce", "olist"], default="ecommerce",
                        help="Dataset to use: 'ecommerce' (original) or 'olist' (Brazilian e-commerce, ~96K customers)")
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
