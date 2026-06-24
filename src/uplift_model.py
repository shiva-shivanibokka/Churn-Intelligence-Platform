"""
Uplift Modeling (Causal ML)
============================
The critical differentiator between a portfolio churn project and
what Netflix, Uber, and Salesforce actually run in production.

Standard churn model question: "Who will churn?"
Uplift model question: "Who will RESPOND to a retention intervention?"

These are fundamentally different optimization problems. A naive churn model
wastes retention budget on:
  - "Sure Things": customers who would stay regardless of intervention
  - "Lost Causes": customers who will churn regardless
  - "Sleeping Dogs": customers who would stay but intervention triggers churn

The only valuable targets are "Persuadables" — customers who:
  1. Have meaningful churn probability
  2. Would actually respond positively to a retention offer

Uber open-sourced CausalML for exactly this use case. Netflix uses uplift
modeling for retention campaign targeting. This implementation uses:
  - S-Learner: single model with treatment as a feature (simpler baseline)
  - T-Learner: separate models for treatment/control groups (standard approach)

Dataset Note on Treatment Assignment:
  The e-commerce dataset does not have a historical A/B test (treatment/control).
  We apply a realistic simulation strategy documented in academic literature:
  - "Complain" flag = proxy for customers who received support outreach (treated)
  - "CouponUsed" > 0 = proxy for customers who received discount offers (treated)
  This is a standard approach for observational uplift modeling when randomized
  experiment data is unavailable. Production systems (Netflix, Uber) use actual
  A/B test logs to train their uplift models.
"""

import logging
import numpy as np
import pandas as pd
import xgboost as xgb
import warnings
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)

try:
    from causalml.inference.meta import BaseSClassifier, BaseTClassifier

    CAUSALML_AVAILABLE = True
except ImportError:
    CAUSALML_AVAILABLE = False
    logger.warning("CausalML not available — using custom T-Learner fallback.")

warnings.filterwarnings("ignore")

MODELS_PATH = os.path.join(os.path.dirname(__file__), "..", "models")
PROCESSED_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


def simulate_treatment(df: pd.DataFrame) -> pd.Series:
    """
    Simulate treatment assignment from observational data.

    Treatment proxy: customers who either filed a complaint (received support outreach)
    OR used a coupon (received a promotional discount offer). This proxies for
    "received some form of retention intervention."

    In a production system, this column would be the actual A/B test assignment
    flag from the experimentation platform (Spotify's 'Confidence', Uber's
    experimentation framework, etc.).
    """
    treatment = ((df["Complain"] == 1) | (df["CouponUsed"] > 0)).astype(int)
    return treatment


def compute_uplift_scores_custom(
    df: pd.DataFrame,
    feature_cols: list,
    treatment_col: str = "Treatment",
    outcome_col: str = "Churn",
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Custom T-Learner uplift implementation (fallback if CausalML unavailable).

    T-Learner: Two separate models.
    - mu_1(x): P(Y=1 | T=1, X=x) — churn probability given treatment
    - mu_0(x): P(Y=1 | T=0, X=x) — churn probability without treatment
    - Uplift(x) = mu_0(x) - mu_1(x)
      (positive = treatment reduces churn probability = persuadable)
    """
    X = df[feature_cols].values
    T = df[treatment_col].values
    Y = df[outcome_col].values

    # Split by treatment group
    mask_t1 = T == 1
    mask_t0 = T == 0

    X_t1, y_t1 = X[mask_t1], Y[mask_t1]
    X_t0, y_t0 = X[mask_t0], Y[mask_t0]

    xgb_params = dict(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric="auc",
        n_jobs=-1,
    )

    # Model for treated group
    clf_t1 = xgb.XGBClassifier(**xgb_params)
    clf_t1.fit(X_t1, y_t1)

    # Model for control group
    clf_t0 = xgb.XGBClassifier(**xgb_params)
    clf_t0.fit(X_t0, y_t0)

    # Predict on all customers under both scenarios
    p_churn_treated = clf_t1.predict_proba(X)[:, 1]  # P(churn | if treated)
    p_churn_control = clf_t0.predict_proba(X)[:, 1]  # P(churn | if not treated)

    # Uplift = reduction in churn probability from treatment
    uplift_scores = p_churn_control - p_churn_treated

    metrics = {
        "method": "T-Learner (custom)",
        "treatment_group_size": int(mask_t1.sum()),
        "control_group_size": int(mask_t0.sum()),
        "treatment_churn_rate": float(y_t1.mean()) if len(y_t1) > 0 else None,
        "control_churn_rate": float(y_t0.mean()) if len(y_t0) > 0 else None,
        "mean_uplift": float(uplift_scores.mean()),
        "std_uplift": float(uplift_scores.std()),
    }

    return uplift_scores, p_churn_control, metrics


def compute_uplift_scores_causalml(
    df: pd.DataFrame,
    feature_cols: list,
    treatment_col: str = "Treatment",
    outcome_col: str = "Churn",
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Uplift modeling using Uber's CausalML library.
    Implements both S-Learner and T-Learner for comparison.
    """
    X = df[feature_cols].values
    T = df[treatment_col].values
    Y = df[outcome_col].values

    base_learner = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        use_label_encoder=False,
        eval_metric="auc",
        random_state=42,
        n_jobs=-1,
    )

    # T-Learner
    t_learner = BaseTClassifier(learner=base_learner)
    t_learner.fit(X, treatment=T, y=Y)
    uplift_t = t_learner.predict(X).flatten()

    # S-Learner (treatment as feature)
    s_learner = BaseSClassifier(learner=base_learner)
    s_learner.fit(X, treatment=T, y=Y)
    uplift_s = s_learner.predict(X).flatten()

    # Average T and S learner scores
    uplift_scores = (uplift_t + uplift_s) / 2.0

    # Counterfactual: churn probability without intervention
    base_learner_0 = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        use_label_encoder=False,
        eval_metric="auc",
        random_state=42,
        n_jobs=-1,
    )
    mask_t0 = T == 0
    if mask_t0.sum() == 0:
        logger.warning("No control-group customers found — ChurnProbNoTreatment set to zero.")
        p_churn_control = np.zeros(len(X))
    else:
        base_learner_0.fit(X[mask_t0], Y[mask_t0])
        p_churn_control = base_learner_0.predict_proba(X)[:, 1]

    metrics = {
        "method": "T-Learner + S-Learner ensemble (CausalML)",
        "treatment_group_size": int((T == 1).sum()),
        "control_group_size": int((T == 0).sum()),
        "treatment_churn_rate": float(Y[T == 1].mean()),
        "control_churn_rate": float(Y[T == 0].mean()),
        "mean_uplift_t": float(uplift_t.mean()),
        "mean_uplift_s": float(uplift_s.mean()),
        "mean_uplift_ensemble": float(uplift_scores.mean()),
        "std_uplift": float(uplift_scores.std()),
    }

    return uplift_scores, p_churn_control, metrics


def classify_customer_type(
    uplift_score: float,
    churn_prob: float,
    uplift_threshold: float = 0.05,
    churn_threshold: float = 0.30,
) -> str:
    """
    Classify customers into the four uplift quadrants.
    Thresholds are configurable — in production these are tuned against
    actual campaign cost and CLV to maximize intervention ROI.

    - Persuadables:   high churn risk + positive uplift → TARGET FOR INTERVENTION
    - Sure Things:    low churn risk + positive uplift → no intervention needed
    - Lost Causes:    high churn risk + negative uplift → waste of budget
    - Sleeping Dogs:  low churn risk + negative uplift → do not disturb
    """
    high_churn = churn_prob >= churn_threshold
    positive_uplift = uplift_score >= uplift_threshold

    if high_churn and positive_uplift:
        return "Persuadable"
    elif not high_churn and positive_uplift:
        return "Sure Thing"
    elif high_churn and not positive_uplift:
        return "Lost Cause"
    else:
        return "Sleeping Dog"


def estimate_intervention_roi(
    df: pd.DataFrame,
    uplift_col: str = "UpliftScore",
    churn_prob_col: str = "ChurnProbability",
    avg_clv: float = 500.0,
    intervention_cost: float = 15.0,
) -> pd.DataFrame:
    """
    Compute expected ROI for retaining each customer via intervention.

    Formula (mirrors Uber and Netflix's intervention cost modeling):
    Expected Retained Value = uplift_score × CLV
    Net ROI = Expected Retained Value − Intervention Cost

    avg_clv: Average Customer Lifetime Value (configurable in UI)
    intervention_cost: Cost of one retention intervention (email=$2, call=$25, discount=$50)
    """
    df = df.copy()
    df["ExpectedRetainedValue"] = df[uplift_col] * avg_clv
    df["InterventionCost"] = intervention_cost
    df["NetROI"] = df["ExpectedRetainedValue"] - intervention_cost
    df["ROIPositive"] = df["NetROI"] > 0

    # Priority rank: Persuadables with highest ROI first
    persuadable_mask = df["CustomerType"] == "Persuadable"
    df["InterventionPriority"] = np.nan
    if persuadable_mask.sum() > 0:
        df.loc[persuadable_mask, "InterventionPriority"] = df.loc[
            persuadable_mask, "NetROI"
        ].rank(ascending=False)

    return df


def run_uplift_pipeline(
    df: pd.DataFrame,
    feature_cols: list,
    avg_clv: float = 500.0,
    intervention_cost: float = 15.0,
) -> dict:
    """
    Full uplift modeling pipeline.
    """
    logger.info("Simulating treatment assignment from observational data...")
    df = df.copy()
    df["Treatment"] = simulate_treatment(df)
    logger.info("Treatment group: %d | Control: %d", df["Treatment"].sum(), (df["Treatment"] == 0).sum())

    if CAUSALML_AVAILABLE:
        logger.info("Running CausalML T-Learner + S-Learner ensemble...")
        uplift_scores, p_churn_control, metrics = compute_uplift_scores_causalml(
            df, feature_cols
        )
    else:
        logger.info("Running custom T-Learner...")
        uplift_scores, p_churn_control, metrics = compute_uplift_scores_custom(
            df, feature_cols
        )

    df["UpliftScore"] = uplift_scores
    df["ChurnProbNoTreatment"] = p_churn_control

    logger.info("Classifying customer types (Persuadable / Sure Thing / Lost Cause / Sleeping Dog)...")
    df["CustomerType"] = df.apply(
        lambda row: classify_customer_type(
            row["UpliftScore"],
            row.get("ChurnProbability", row["ChurnProbNoTreatment"]),
        ),
        axis=1,
    )

    logger.info("Computing intervention ROI...")
    df = estimate_intervention_roi(
        df,
        avg_clv=avg_clv,
        intervention_cost=intervention_cost,
    )

    customer_type_counts = df["CustomerType"].value_counts().to_dict()
    n_persuadable = customer_type_counts.get("Persuadable", 0)
    n_positive_roi = df["ROIPositive"].sum()

    logger.info("Customer types: %s", customer_type_counts)
    logger.info("Persuadables: %d | Positive ROI interventions: %d", n_persuadable, n_positive_roi)
    logger.info("Uplift metrics: %s", metrics)

    # Save
    df.to_parquet(os.path.join(PROCESSED_PATH, "uplift.parquet"), index=False)
    joblib.dump(metrics, os.path.join(MODELS_PATH, "uplift_metrics.pkl"))

    return {
        "df": df,
        "metrics": metrics,
        "customer_type_counts": customer_type_counts,
    }


if __name__ == "__main__":
    import sys

    sys.path.insert(0, os.path.dirname(__file__))

    df = pd.read_parquet(os.path.join(PROCESSED_PATH, "scored.parquet"))
    from features import get_feature_sets

    feature_sets = get_feature_sets()

    results = run_uplift_pipeline(df, feature_sets["uplift_model"])
    df_out = results["df"]

    print("\nCustomer Type Distribution:")
    print(df_out["CustomerType"].value_counts())
    print("\nTop 10 Persuadables by ROI:")
    persuadables = df_out[df_out["CustomerType"] == "Persuadable"].nlargest(
        10, "NetROI"
    )
    print(
        persuadables[
            ["CustomerID", "Segment", "ChurnProbability", "UpliftScore", "NetROI"]
        ].to_string()
    )
