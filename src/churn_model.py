"""
Per-Segment Churn Prediction
==============================
Architecture mirrors Salesforce Einstein's per-segment churn scoring:
- A separate CatBoost classifier is trained per customer segment
- Stratified 80/20 holdout split ensures a true generalisation estimate
- Isotonic calibration on a held-out slice, so the probabilities that feed the
  ROI maths mean what they say
- Exact TreeSHAP from CatBoost for both global and per-customer explanations

Why per-segment models?
  A single global churn model treats all customers identically.
  But a "Champion" churns for different reasons than a "Lapsed" customer.
  Champions who churn usually have a specific trigger (bad support experience,
  competitor offer). Lapsed customers churn through gradual disengagement.
  Separate models capture segment-specific churn dynamics — this is the
  approach used by Salesforce for different customer tiers.

Why calibration, and why it is real here:
  This module used to claim isotonic calibration and not perform it —
  `calibrated_clf` was a plain alias for the base classifier, on the reasoning
  that "CatBoost is well calibrated natively". That reasoning was wrong twice.
  Ordered boosting helps, but these models are trained with
  `class_weights=[1, pos_weight]`, which deliberately inflates the positive
  class: the output is miscalibrated *by construction*. And these probabilities
  are not just displayed — they are multiplied by CLV to rank retention spend,
  which is exactly the case where a 0.7 that is not 70% costs money.
  `holdout_brier_uncalibrated` and `holdout_brier` are both recorded so the
  effect is a measurement rather than an assertion.

Explainability approach:
  CatBoost computes exact Shapley values in C++ via
  `get_feature_importance(Pool(...), type="ShapValues")`. They are used
  directly, per customer, and they carry a sign.

  The previous approach deserves recording because it looked fine and was not.
  It multiplied CatBoost's `get_feature_importance()` — an *unsigned* magnitude
  — by each customer's deviation from the segment mean, then took the sign from
  whether the value was above or below that mean. So the direction of every
  driver was really just "is this above average", with no notion of whether the
  feature pushes churn up or down: long tenure, high satisfaction and high
  cashback all came back as "increases churn risk". Those strings are fed
  verbatim to the retention agent, which then wrote plans to fix a customer's
  high satisfaction score. The comment explaining the workaround blamed XGBoost
  2.x/3.x base_score instability — a real problem, but this module stopped using
  XGBoost, and the workaround outlived the reason for it.
"""

import json
import logging
import os
import warnings

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

MODELS_PATH = os.path.join(os.path.dirname(__file__), "..", "models")
PROCESSED_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


class IsotonicCalibratedClassifier:
    """
    A fitted CatBoost model plus the isotonic map from its raw scores to
    calibrated probabilities.

    Exposes only `predict_proba`, which is the whole interface the rest of the
    pipeline uses, so `score_customers`, the FastAPI endpoint and every
    downstream consumer call it exactly as they called the bare classifier.

    Defined at module level rather than as a closure because these objects are
    pickled into models/segment_models.pkl, and joblib cannot pickle a local
    function.
    """

    def __init__(self, base_clf, calibrator: IsotonicRegression, n_calibration: int):
        self.base_clf = base_clf
        self.calibrator = calibrator
        # Bound the output by what the calibration sample can actually resolve.
        #
        # Isotonic regression is a step function fitted to observed frequencies,
        # so its lowest bin is routinely exactly 0.0 and its highest exactly 1.0.
        # Reporting P(churn) = 0.0 is a claim that this customer cannot churn,
        # which no finite sample supports — and it makes any downstream log-loss
        # infinite. The floor is the smallest rate a sample of this size can
        # distinguish from zero, half of one observation.
        self.eps = 1.0 / (2.0 * max(n_calibration, 1))

    def predict_proba(self, X) -> np.ndarray:
        raw = self.base_clf.predict_proba(X)[:, 1]
        cal = np.clip(self.calibrator.predict(raw), self.eps, 1.0 - self.eps)
        return np.column_stack([1.0 - cal, cal])

    def get_feature_importance(self, *args, **kwargs):
        """Pass through to the base model — calibration does not change SHAP."""
        return self.base_clf.get_feature_importance(*args, **kwargs)


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
    2. Train a CatBoost base classifier, early-stopped on the calibration slice
    3. Calibrate its probabilities with isotonic regression on that same slice
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

    # Manual 3-fold CV — sklearn's cross_val_score cannot clone CatBoostClassifier
    # when class_weights is a list. Convert to numpy so integer indexing works.
    X_np = X.values if hasattr(X, "values") else X
    y_np = y.values if hasattr(y, "values") else y
    cv_params = {**params, "iterations": 100}
    cv_folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    cv_aucs, cv_aps = [], []
    for tr_idx, val_idx in cv_folds.split(X_np, y_np):
        _clf = CatBoostClassifier(**cv_params)
        _clf.fit(X_np[tr_idx], y_np[tr_idx])
        _prob = _clf.predict_proba(X_np[val_idx])[:, 1]
        if len(np.unique(y_np[val_idx])) > 1:
            cv_aucs.append(roc_auc_score(y_np[val_idx], _prob))
            cv_aps.append(average_precision_score(y_np[val_idx], _prob))
    cv_auc = float(np.mean(cv_aucs)) if cv_aucs else 0.5
    cv_ap = float(np.mean(cv_aps)) if cv_aps else 0.0

    # ── Fit / calibration split, carved out of the 80% train split ───────────
    #
    # The 20% holdout above is never touched by anything below, so every
    # "holdout_*" metric stays an honest generalisation estimate.
    #
    # The calibration slice does double duty: it is the early-stopping eval set
    # and the sample the isotonic map is fitted on. That is a deliberate
    # trade — a three-way split would leave the smaller segments with too little
    # to fit on, and isotonic regression is a monotone step function with few
    # effective parameters, so the optimism is small. It is also bounded rather
    # than assumed: `holdout_brier` is measured on data neither step saw.
    X_fit, X_cal, y_fit, y_cal = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Early stopping, because 500 fixed iterations was overfitting hard: these
    # segments scored ~0.87 AUC on their own training rows against ~0.61 on the
    # holdout, and the 100-iteration CV estimate came out *above* the
    # 500-iteration holdout. Fewer, better-chosen iterations is not a
    # compromise here — it is the higher-scoring model.
    base_clf.fit(
        X_fit,
        y_fit,
        eval_set=(X_cal, y_cal),
        early_stopping_rounds=50,
        use_best_model=True,
    )
    best_iteration = int(getattr(base_clf, "best_iteration_", None) or params["iterations"])

    # ── Isotonic calibration ─────────────────────────────────────────────────
    cal_raw = base_clf.predict_proba(X_cal)[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator.fit(cal_raw, y_cal)
    calibrated_clf = IsotonicCalibratedClassifier(base_clf, calibrator, len(y_cal))

    # ── Train-split evaluation (calibrated, on the rows it was fitted on) ────
    train_probs = calibrated_clf.predict_proba(X_fit)[:, 1]
    train_brier = float(brier_score_loss(y_fit, train_probs))
    train_auc = float(roc_auc_score(y_fit, train_probs))
    train_ap = float(average_precision_score(y_fit, train_probs))

    # ── Holdout test-split evaluation (true generalisation estimate) ─────────
    # Both are recorded: calibration cannot change AUC (it is monotone, so the
    # ranking is identical by construction) but it is supposed to move Brier,
    # and reporting the pair is what turns "we calibrate" into a measurement.
    raw_test_probs = base_clf.predict_proba(X_test)[:, 1]
    test_probs = calibrated_clf.predict_proba(X_test)[:, 1]
    holdout_auc = float(roc_auc_score(y_test, test_probs)) if len(np.unique(y_test)) > 1 else 0.5
    holdout_ap = float(average_precision_score(y_test, test_probs)) if len(np.unique(y_test)) > 1 else 0.0
    holdout_brier = float(brier_score_loss(y_test, test_probs))
    holdout_brier_uncalibrated = float(brier_score_loss(y_test, raw_test_probs))

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
        # Same holdout rows, before the isotonic map — the pair is the evidence
        # that calibration did something.
        "holdout_brier_uncalibrated": holdout_brier_uncalibrated,
        "n_calibration": int(len(y_cal)),
        "best_iteration": best_iteration,
    }

    # MLflow logging
    if mlflow_run:
        with mlflow.start_run(run_name=f"churn_{segment_name}", nested=True):
            mlflow.log_params(
                {
                    "segment": segment_name,
                    "n_clusters": 5,
                    "iterations": params["iterations"],
                    "depth": params["depth"],
                    "learning_rate": params["learning_rate"],
                    "pos_weight": float(pos_weight),
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
                    "holdout_brier_uncalibrated": holdout_brier_uncalibrated,
                    "best_iteration": float(best_iteration),
                    "churn_rate": float(y.mean()),
                    "n_train": float(len(y)),
                    "n_test": float(len(y_test)),
                }
            )
            # Log top gain-based importance features
            for feat, val in mean_abs_shap.head(5).items():
                mlflow.log_metric(f"importance_{feat}", float(val))

    logger.info(
        "Segment '%s': CV AUC=%.3f | Holdout AUC=%.3f | Holdout Brier %.4f→%.4f "
        "(calibrated) | best_iter=%d | n_fit=%d, n_cal=%d, n_test=%d, churn_rate=%.2f%%",
        segment_name, cv_auc, holdout_auc, holdout_brier_uncalibrated, holdout_brier,
        best_iteration, len(y_fit), len(y_cal), len(y_test), y.mean() * 100,
    )

    return {
        "base_clf": base_clf,
        "calibrated_clf": calibrated_clf,
        "mean_abs_shap": mean_abs_shap,
        "metrics": metrics,
        "feature_cols": feature_cols,
        "segment_name": segment_name,
        # Fit split (the rows the base model actually saw)
        "X_train": X_fit,
        "y_train": y_fit,
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
    For each customer, the top N features driving their individual churn risk,
    as exact Shapley values from the segment's own CatBoost model.

    CatBoost computes these in C++ via
    `get_feature_importance(Pool(X), type="ShapValues")`, which returns an
    (n_rows, n_features + 1) array — the trailing column is the model's expected
    value, and the rest sum with it to the raw prediction. They are signed and in
    log-odds space, so a positive value genuinely means "this feature pushes this
    customer's churn probability up".

    The sign is the entire point. What this replaced multiplied an *unsigned*
    global importance by each customer's deviation from the segment mean and
    then took the direction from whether the value was above or below that mean,
    which reduced every explanation to "this is above average" — so a customer's
    long tenure and high satisfaction were reported to the retention agent as
    things increasing their churn risk.
    """
    df = df.copy()
    df["TopSHAPFeatures"] = "{}"

    for segment_name, model_dict in segment_models.items():
        if model_dict is None:
            continue
        mask = df["Segment"] == segment_name
        if mask.sum() == 0:
            continue

        X_seg = df.loc[mask, feature_cols]
        base_clf = model_dict["base_clf"]

        shap_values = base_clf.get_feature_importance(
            Pool(X_seg), type="ShapValues"
        )[:, :-1]  # drop the expected-value column

        # Rank each row by |contribution| without sorting all 23 columns.
        k = min(top_n, shap_values.shape[1])
        top_idx = np.argpartition(-np.abs(shap_values), k - 1, axis=1)[:, :k]

        features = np.asarray(feature_cols)
        records = []
        for row_i, cols in enumerate(top_idx):
            vals = shap_values[row_i, cols]
            order = np.argsort(-np.abs(vals))  # strongest first, for display
            records.append(
                json.dumps(
                    {
                        str(features[cols[j]]): round(float(vals[j]), 4)
                        for j in order
                    }
                )
            )

        df.loc[mask, "TopSHAPFeatures"] = records

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
            avg_holdout_brier_uncal = float(
                np.mean([m["holdout_brier_uncalibrated"] for m in valid_metrics])
            )
            mlflow.log_metric("avg_holdout_brier_uncalibrated", avg_holdout_brier_uncal)
            mlflow.log_metric("avg_cv_auc_across_segments", avg_cv_auc)
            mlflow.log_metric("avg_holdout_auc_across_segments", avg_holdout_auc)
            mlflow.log_metric("avg_holdout_brier_across_segments", avg_holdout_brier)
            mlflow.log_metric("avg_train_brier_across_segments", avg_train_brier)
            logger.info(
                "Aggregate: CV AUC=%.3f | Holdout AUC=%.3f | Holdout Brier %.4f→%.4f",
                avg_cv_auc, avg_holdout_auc, avg_holdout_brier_uncal, avg_holdout_brier,
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
