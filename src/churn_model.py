"""
Per-Segment Churn Prediction
==============================
Architecture mirrors Salesforce Einstein's per-segment churn scoring:
- A separate XGBoost classifier is trained per customer segment
- Stratified 80/20 holdout split ensures a true generalisation estimate
- Probability calibration with isotonic regression ensures raw model
  probabilities are reliable for ROI calculations (CLV, retention budget)
- XGBoost gain-based feature importance provides global segment-level
  explainability; per-customer explanations use a deviation-weighted
  approximation combining global importance with individual feature values

Why per-segment models?
  A single global churn model treats all customers identically.
  But a "Champion" churns for different reasons than a "Lapsed" customer.
  Champions who churn usually have a specific trigger (bad support experience,
  competitor offer). Lapsed customers churn through gradual disengagement.
  Separate models capture segment-specific churn dynamics — this is the
  approach used by Salesforce for different customer tiers.

Why calibration?
  Raw XGBoost probabilities are not well-calibrated — a 0.7 probability
  does not mean 70% of customers at that score actually churn.
  Calibration is required whenever probabilities are used in business
  calculations (CLV, retention ROI, budget allocation). Isotonic regression
  is preferred over Platt scaling for non-parametric data.

Explainability approach:
  TreeExplainer-based SHAP interaction values are not used because
  XGBoost 2.x/3.x changes to base_score handling introduce instability
  in interaction value computation. Instead this module uses:
    1. Global: XGBoost gain-based feature importance (normalised to [0,1])
    2. Per-customer: deviation from segment mean, weighted by global importance
  This approximation is fast, stable, and sufficient for the retention
  team's use case (rank features by contribution, not compute Shapley values).
  Full TreeExplainer can be re-enabled once XGBoost stabilises the API.
"""

import logging
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
import os
import json
import warnings
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

MODELS_PATH = os.path.join(os.path.dirname(__file__), "..", "models")
PROCESSED_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


def get_catboost_params(pos_weight: float = 5.0) -> dict:
    """
    CatBoost hyperparameters tuned for imbalanced churn datasets.

    Why CatBoost over XGBoost:
    - Handles categorical features natively (no label encoding needed, though we still
      pass integers — CatBoost works fine either way)
    - Ordered boosting reduces overfitting on small segments without explicit subsampling
    - scale_pos_weight equivalent: class_weights = {0: 1, 1: pos_weight}
    - No need for use_label_encoder or eval_metric hacks
    """
    return {
        "iterations": 500,
        "depth": 6,
        "learning_rate": 0.05,
        "l2_leaf_reg": 3.0,
        "random_strength": 1.0,
        "bagging_temperature": 0.5,
        "border_count": 128,
        "class_weights": [1.0, pos_weight],
        "eval_metric": "AUC",
        "random_seed": 42,
        "verbose": 0,
        "thread_count": -1,
    }


def train_segment_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    segment_name: str,
    feature_cols: list,
    mlflow_run: bool = True,
) -> dict:
    """
    Train and calibrate a single segment's churn model.

    Steps:
    1. Hold out 20% of the segment as a true test set (stratified split)
    2. Train XGBoost base classifier on the 80% train split
    3. Calibrate probabilities with isotonic regression
    4. Compute cross-validated AUC on train split (variance estimate)
    5. Compute holdout AUC/AP/Brier on the 20% test split (generalisation estimate)
    6. Log all metrics and the model to MLflow

    Returns a dict with model, calibrated model, train metrics, and holdout metrics.

    Why separate CV and holdout?
      CV AUC on training data measures model capacity but overestimates
      generalisation — it never tests on data the model was fitted on,
      but it was still used to select the hyperparameters. The holdout
      test set is completely unseen: it gives the true generalisation estimate
      reported in the README and to stakeholders.
    """
    X_all = X_train[feature_cols]
    y_all = y_train

    # Skip segments with too few samples or only one class
    if len(y_all) < 50 or y_all.nunique() < 2:
        logger.warning("Skipping segment '%s': insufficient data (%d rows)", segment_name, len(y_all))
        return None

    # ── Stratified 80/20 holdout split ──────────────────────────────────────
    # Stratify on y to preserve churn rate in both splits.
    # random_state=42 ensures reproducible splits across runs.
    X, X_test, y, y_test = train_test_split(
        X_all, y_all, test_size=0.20, random_state=42, stratify=y_all
    )

    # Recompute class weight per segment (churn rates differ by segment)
    neg, pos = (y == 0).sum(), (y == 1).sum()
    pos_weight = max(1.0, neg / pos) if pos > 0 else 5.0
    params = get_catboost_params(pos_weight)

    base_clf = CatBoostClassifier(**params)

    # Cross-validated metrics on the TRAIN split only (5-fold stratified)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_auc = float(cross_val_score(base_clf, X, y, cv=cv, scoring="roc_auc").mean())
    cv_ap = float(
        cross_val_score(base_clf, X, y, cv=cv, scoring="average_precision").mean()
    )

    # Fit base model on 80% of the train split; reserve 20% for calibration.
    # Calibrating on the same data the model trained on inflates calibration
    # quality (the model has already memorised those labels).
    X_fit, X_calib, y_fit, y_calib = train_test_split(
        X, y, test_size=0.20, random_state=123, stratify=y
    )
    base_clf.fit(X_fit, y_fit)

    calibrated_clf = CalibratedClassifierCV(base_clf, cv="prefit", method="isotonic")
    calibrated_clf.fit(X_calib, y_calib)

    # ── Train-split evaluation (calibrated model) ────────────────────────────
    train_probs = calibrated_clf.predict_proba(X)[:, 1]
    train_brier = float(brier_score_loss(y, train_probs))
    train_auc = float(roc_auc_score(y, train_probs))
    train_ap = float(average_precision_score(y, train_probs))

    # ── Holdout test-split evaluation (true generalisation estimate) ─────────
    # This is the metric reported in the README.
    # The model has never seen X_test during training or calibration.
    test_probs = calibrated_clf.predict_proba(X_test)[:, 1]
    holdout_auc = float(roc_auc_score(y_test, test_probs))
    holdout_ap = float(average_precision_score(y_test, test_probs))
    holdout_brier = float(brier_score_loss(y_test, test_probs))

    # Convenience alias so callers keep using probs/brier as before
    probs = train_probs
    brier = train_brier

    # CatBoost native feature importance (PredictionValuesChange — equivalent to XGBoost gain).
    # Returns a numpy array aligned to feature_cols order.
    importance_arr = base_clf.get_feature_importance()
    mean_abs_shap = pd.Series(importance_arr, index=feature_cols).sort_values(ascending=False)

    # Normalize to [0,1] range for interpretability
    max_val = mean_abs_shap.max()
    if max_val > 0:
        mean_abs_shap = mean_abs_shap / max_val

    metrics = {
        "segment": segment_name,
        # Train split size (80% of segment)
        "n_train": int(len(y)),
        "n_test": int(len(y_test)),
        "n_churners_train": int(y.sum()),
        "churn_rate_train": float(y.mean()),
        # Cross-validation on train split (variance estimate — lower bias than single split)
        "cv_auc": cv_auc,
        "cv_ap": cv_ap,
        # Train-split evaluation (calibrated model)
        "train_auc": train_auc,
        "train_ap": train_ap,
        "train_brier": train_brier,
        # Holdout test-split evaluation (TRUE generalisation estimate — report this)
        "holdout_auc": holdout_auc,
        "holdout_ap": holdout_ap,
        "holdout_brier": holdout_brier,
    }

    # MLflow logging
    if mlflow_run:
        with mlflow.start_run(run_name=f"churn_{segment_name}", nested=True):
            mlflow.log_params(
                {
                    "segment": segment_name,
                    "n_clusters": 5,
                    "n_estimators": params["n_estimators"],
                    "max_depth": params["max_depth"],
                    "learning_rate": params["learning_rate"],
                    "scale_pos_weight": float(params["scale_pos_weight"]),
                    "calibration_method": "isotonic",
                    "holdout_pct": 0.20,
                }
            )
            mlflow.log_metrics(
                {
                    "cv_auc": cv_auc,
                    "cv_ap": cv_ap,
                    "train_auc": train_auc,
                    "train_ap": train_ap,
                    "train_brier": train_brier,
                    # Holdout metrics — these are what matter for reporting
                    "holdout_auc": holdout_auc,
                    "holdout_ap": holdout_ap,
                    "holdout_brier": holdout_brier,
                    "churn_rate": float(y.mean()),
                    "n_train": float(len(y)),
                    "n_test": float(len(y_test)),
                }
            )
            # Log top gain-based importance features
            for feat, val in mean_abs_shap.head(5).items():
                mlflow.log_metric(f"importance_{feat}", float(val))

            mlflow.sklearn.log_model(base_clf, name=f"model_{segment_name}")

    logger.info(
        "Segment '%s': CV AUC=%.3f | Holdout AUC=%.3f | Holdout Brier=%.3f | "
        "n_train=%d, n_test=%d, churn_rate=%.2f%%",
        segment_name, cv_auc, holdout_auc, holdout_brier, len(y), len(y_test), y.mean() * 100,
    )

    return {
        "base_clf": base_clf,
        "calibrated_clf": calibrated_clf,
        "mean_abs_shap": mean_abs_shap,
        "metrics": metrics,
        "feature_cols": feature_cols,
        "segment_name": segment_name,
        # Train split (used for per-customer SHAP approximation)
        "X_train": X,
        "y_train": y,
        # Holdout split (kept for post-hoc analysis and bias checks)
        "X_test": X_test,
        "y_test": y_test,
    }


def score_customers(
    df: pd.DataFrame, segment_models: dict, feature_cols: list
) -> pd.DataFrame:
    """
    Score all customers with their segment-specific calibrated churn probabilities.

    Each customer is scored using the model trained on their segment.
    This avoids the global model bias where the same features mean
    different things for Champions vs. Lapsed customers.
    """
    df = df.copy()
    df["ChurnProbability"] = np.nan
    df["ChurnPrediction"] = np.nan

    for segment_name, model_dict in segment_models.items():
        if model_dict is None:
            continue
        mask = df["Segment"] == segment_name
        if mask.sum() == 0:
            continue

        X_seg = df.loc[mask, feature_cols]
        probs = model_dict["calibrated_clf"].predict_proba(X_seg)[:, 1]
        preds = (probs >= 0.5).astype(int)
        df.loc[mask, "ChurnProbability"] = probs
        df.loc[mask, "ChurnPrediction"] = preds

    # Risk tier labeling (mirrors Salesforce's health score tiers: Red/Yellow/Green)
    df["RiskTier"] = pd.cut(
        df["ChurnProbability"],
        bins=[0, 0.3, 0.6, 1.0],
        labels=["Low Risk", "Medium Risk", "High Risk"],
        include_lowest=True,
    )

    return df


def compute_per_customer_shap(
    df: pd.DataFrame,
    segment_models: dict,
    feature_cols: list,
    top_n: int = 5,
) -> pd.DataFrame:
    """
    For each customer, compute the top N features driving their individual churn risk.

    Method: deviation-weighted importance approximation
    -------------------------------------------------------
    For each customer we combine two signals:
      1. Global feature importance (XGBoost gain-based, normalised to [0,1])
         — which features matter most for this SEGMENT
      2. Per-customer deviation from the segment mean (z-scored)
         — which features are most abnormal for THIS individual

    weighted_score[feature] = |z_score[feature]| × global_importance[feature]

    The sign is set by whether the feature value is above or below the segment
    mean (above = increases churn risk for a positively-important feature).

    This gives actionable, human-readable explanations in the Streamlit UI
    without the latency and instability of running TreeExplainer at inference
    time. The approximation is sufficient for the retention team's use case:
    understanding WHY a specific customer is flagged as high-risk.
    """
    df = df.copy()
    shap_records = []

    for segment_name, model_dict in segment_models.items():
        if model_dict is None:
            continue
        mask = df["Segment"] == segment_name
        if mask.sum() == 0:
            continue

        X_seg = df.loc[mask, feature_cols]
        base_clf = model_dict["base_clf"]

        # Global SHAP importance for this segment (already computed, very fast)
        global_shap = model_dict["mean_abs_shap"]  # pd.Series indexed by feature name

        # Per-customer: multiply global SHAP weight by deviation from segment mean
        # This gives a fast approximation of which features are driving THIS customer's risk
        seg_mean = X_seg.mean()
        seg_std = X_seg.std().replace(0, 1)  # avoid division by zero

        for idx, row in X_seg.iterrows():
            # Standardized deviation from segment mean
            deviation = (row - seg_mean).abs() / seg_std

            # Weight deviation by global SHAP importance
            weighted = deviation * global_shap.reindex(feature_cols).fillna(0)

            # Top N features by weighted score
            top_feats = weighted.nlargest(top_n)

            # Assign sign: positive if above mean (increases risk for risk features)
            top_dict = {}
            for feat in top_feats.index:
                raw_val = float(row[feat])
                mean_val = float(seg_mean[feat])
                shap_sign = float(global_shap.get(feat, 0))
                # Approximate signed SHAP: positive = above mean for a positive-SHAP feature
                signed = abs(float(weighted[feat])) * (1 if raw_val > mean_val else -1)
                top_dict[feat] = round(signed, 4)

            shap_records.append(
                {
                    "CustomerID": df.loc[idx, "CustomerID"]
                    if "CustomerID" in df.columns
                    else idx,
                    "index": idx,
                    "TopSHAPFeatures": json.dumps(top_dict),
                }
            )

    if not shap_records:
        df["TopSHAPFeatures"] = "{}"
        return df

    shap_df = pd.DataFrame(shap_records).set_index("index")
    df = df.join(shap_df[["TopSHAPFeatures"]], how="left", rsuffix="_shap")
    return df


def run_churn_pipeline(
    df: pd.DataFrame,
    feature_cols: list,
    experiment_name: str = "CustomerChurnEngine",
) -> dict:
    """
    Full per-segment churn modeling pipeline with MLflow tracking.
    """
    os.makedirs(MODELS_PATH, exist_ok=True)

    mlflow.set_experiment(experiment_name)

    segment_models = {}
    all_metrics = []

    segments = df["Segment"].unique()
    logger.info("Training per-segment models for %d segments...", len(segments))

    with mlflow.start_run(run_name="PerSegmentChurnPipeline"):
        mlflow.log_param("n_segments", len(segments))
        mlflow.log_param("feature_count", len(feature_cols))
        mlflow.log_param("total_customers", len(df))
        mlflow.log_param("global_churn_rate", float(df["Churn"].mean()))

        for segment in segments:
            mask = df["Segment"] == segment
            X_seg = df.loc[mask]
            y_seg = df.loc[mask, "Churn"]

            model_dict = train_segment_model(
                X_seg, y_seg, segment, feature_cols, mlflow_run=True
            )
            segment_models[segment] = model_dict
            if model_dict:
                all_metrics.append(model_dict["metrics"])

        # Log aggregate metrics (both CV and holdout — report holdout as the headline)
        valid_metrics = [m for m in all_metrics if m]
        if valid_metrics:
            avg_cv_auc = float(np.mean([m["cv_auc"] for m in valid_metrics]))
            avg_holdout_auc = float(np.mean([m["holdout_auc"] for m in valid_metrics]))
            avg_holdout_brier = float(
                np.mean([m["holdout_brier"] for m in valid_metrics])
            )
            avg_train_brier = float(np.mean([m["train_brier"] for m in valid_metrics]))
            mlflow.log_metric("avg_cv_auc_across_segments", avg_cv_auc)
            mlflow.log_metric("avg_holdout_auc_across_segments", avg_holdout_auc)
            mlflow.log_metric("avg_holdout_brier_across_segments", avg_holdout_brier)
            mlflow.log_metric("avg_train_brier_across_segments", avg_train_brier)
            logger.info(
                "Aggregate: CV AUC=%.3f | Holdout AUC=%.3f | Holdout Brier=%.3f",
                avg_cv_auc, avg_holdout_auc, avg_holdout_brier,
            )

    # Score all customers
    logger.info("Scoring all customers with calibrated probabilities...")
    df_scored = score_customers(df, segment_models, feature_cols)

    # Per-customer SHAP (top features)
    logger.info("Computing per-customer SHAP explanations...")
    df_scored = compute_per_customer_shap(df_scored, segment_models, feature_cols)

    # Save artifacts
    joblib.dump(segment_models, os.path.join(MODELS_PATH, "segment_models.pkl"))
    df_scored.to_parquet(os.path.join(PROCESSED_PATH, "scored.parquet"), index=False)
    logger.info("Saved scored data. High-risk customers: %d", (df_scored["RiskTier"] == "High Risk").sum())

    return {
        "df": df_scored,
        "segment_models": segment_models,
        "metrics": all_metrics,
    }


if __name__ == "__main__":
    import sys

    sys.path.insert(0, os.path.dirname(__file__))
    from features import build_pipeline, get_feature_sets
    from segmentation import run_segmentation

    df = build_pipeline(save=True)
    feature_sets = get_feature_sets()

    seg_results = run_segmentation(df, feature_sets["clustering"])
    df_seg = seg_results["df"]

    churn_results = run_churn_pipeline(df_seg, feature_sets["churn_model"])
    df_final = churn_results["df"]

    print("\nRisk Distribution:")
    print(df_final["RiskTier"].value_counts())
    print("\nChurn Rate by Segment:")
    print(df_final.groupby("Segment")["ChurnProbability"].mean().round(3))
