"""Tests for churn model scoring and classification logic."""

import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from churn_model import score_customers
from uplift_model import classify_customer_type


class TestClassifyCustomerType:
    """Unit tests for the four-quadrant uplift classification."""

    def test_persuadable(self):
        assert classify_customer_type(uplift_score=0.10, churn_prob=0.50) == "Persuadable"

    def test_sure_thing(self):
        assert classify_customer_type(uplift_score=0.10, churn_prob=0.10) == "Sure Thing"

    def test_lost_cause(self):
        assert classify_customer_type(uplift_score=-0.10, churn_prob=0.50) == "Lost Cause"

    def test_sleeping_dog(self):
        assert classify_customer_type(uplift_score=-0.10, churn_prob=0.10) == "Sleeping Dog"

    def test_boundary_churn_threshold(self):
        # Just below churn threshold (0.30 uses >=, so 0.29 is not high churn)
        result = classify_customer_type(uplift_score=0.10, churn_prob=0.29)
        assert result == "Sure Thing"

    def test_boundary_uplift_threshold(self):
        # Exactly at uplift threshold (0.05) — should be "positive uplift"
        result = classify_customer_type(uplift_score=0.05, churn_prob=0.50)
        assert result == "Persuadable"

    def test_custom_thresholds(self):
        result = classify_customer_type(
            uplift_score=0.03, churn_prob=0.50,
            uplift_threshold=0.02, churn_threshold=0.40,
        )
        assert result == "Persuadable"


class TestScoreCustomers:
    """Tests for score_customers using a mock model dict."""

    def _make_mock_segment_models(self):
        """Return a minimal mock that mimics the structure returned by train_segment_model."""
        from unittest.mock import MagicMock
        mock_clf = MagicMock()
        mock_clf.predict_proba.return_value = np.array([[0.4, 0.6]])
        return {
            "Champions": {
                "calibrated_clf": mock_clf,
                "feature_cols": ["EngagementScore", "RecencySignal"],
            }
        }

    def test_churn_probability_column_added(self):
        models = self._make_mock_segment_models()
        # Build a minimal one-row df
        df = pd.DataFrame({
            "Segment": ["Champions"],
            "EngagementScore": [0.7],
            "RecencySignal": [0.3],
        })
        result = score_customers(df, models, ["EngagementScore", "RecencySignal"])
        assert "ChurnProbability" in result.columns

    def test_risk_tier_assigned(self):
        models = self._make_mock_segment_models()
        df = pd.DataFrame({
            "Segment": ["Champions"],
            "EngagementScore": [0.7],
            "RecencySignal": [0.3],
        })
        result = score_customers(df, models, ["EngagementScore", "RecencySignal"])
        assert "RiskTier" in result.columns
        assert result["RiskTier"].iloc[0] in ["Low Risk", "Medium Risk", "High Risk"]

    def test_high_prob_maps_to_high_risk(self):
        from unittest.mock import MagicMock
        mock_clf = MagicMock()
        mock_clf.predict_proba.return_value = np.array([[0.1, 0.9]])  # 90% churn
        models = {"At-Risk": {"calibrated_clf": mock_clf, "feature_cols": ["EngagementScore"]}}
        df = pd.DataFrame({"Segment": ["At-Risk"], "EngagementScore": [0.1]})
        result = score_customers(df, models, ["EngagementScore"])
        assert result["RiskTier"].iloc[0] == "High Risk"


class TestCalibration:
    """
    `calibrated_clf` was a plain alias for `base_clf` — the README, the
    architecture doc and the skills list all led with isotonic calibration and
    the code performed none. It was hidden by naming: every caller used
    `model_dict["calibrated_clf"]`, so the call sites read as calibrated
    regardless.

    These assert the property rather than the presence of the word.
    """

    def _artifacts(self):
        import joblib
        path = os.path.join(os.path.dirname(__file__), "..", "models", "segment_models.pkl")
        if not os.path.exists(path):
            pytest.skip("segment_models.pkl not present — run python src/pipeline.py first")
        return joblib.load(path)

    def test_calibrated_model_is_not_the_base_model(self):
        for segment, md in self._artifacts().items():
            if md is None:
                continue
            assert md["calibrated_clf"] is not md["base_clf"], (
                f"Segment '{segment}' ships an uncalibrated model under the name "
                "calibrated_clf."
            )

    def test_calibration_improves_brier_on_held_out_data(self):
        """
        The claim is that calibration helps, so the pair of numbers is recorded
        and compared. Averaged across segments rather than required per-segment:
        isotonic can lose slightly on a small, already well-ordered segment, and
        demanding a win everywhere would invite tuning until it appeared.
        """
        metrics = [md["metrics"] for md in self._artifacts().values() if md]
        raw = np.mean([m["holdout_brier_uncalibrated"] for m in metrics])
        cal = np.mean([m["holdout_brier"] for m in metrics])
        assert cal < raw, f"calibration made Brier worse: {raw:.4f} -> {cal:.4f}"

    def test_probabilities_are_never_exactly_zero_or_one(self):
        """
        Isotonic's end bins are literally 0.0 and 1.0. Reporting those is a claim
        of certainty no finite sample supports, and a 0.0 makes any downstream
        log-loss infinite.
        """
        for segment, md in self._artifacts().items():
            if md is None:
                continue
            probs = md["calibrated_clf"].predict_proba(md["X_test"])[:, 1]
            assert probs.min() > 0.0, f"{segment} emitted P(churn) = 0"
            assert probs.max() < 1.0, f"{segment} emitted P(churn) = 1"

    def test_mean_prediction_tracks_the_base_rate(self):
        """
        The headline property of a calibrated model: predicted risk summed over
        a population should match how many actually churned. An uncalibrated
        model trained with class_weights overstates it by construction.
        """
        path = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "uplift.parquet")
        if not os.path.exists(path):
            pytest.skip("uplift.parquet not present — run python src/pipeline.py first")
        df = pd.read_parquet(path)
        assert df["ChurnProbability"].mean() == pytest.approx(df["Churn"].mean(), abs=0.02)


class TestPerCustomerShap:
    """
    The old explanations took their sign from "is this value above the segment
    mean", weighted by an *unsigned* importance — so long tenure and high
    satisfaction were reported as things increasing a customer's churn risk, and
    those strings went straight into the retention agent's prompt.
    """

    def _shap_frame(self):
        path = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "uplift.parquet")
        if not os.path.exists(path):
            pytest.skip("uplift.parquet not present — run python src/pipeline.py first")
        return pd.read_parquet(path)

    def test_every_customer_has_explanations(self):
        df = self._shap_frame()
        parsed = df["TopSHAPFeatures"].map(json.loads)
        assert (parsed.map(len) > 0).all()

    def test_signs_are_not_determined_by_being_above_the_mean(self):
        """
        Under the old scheme a feature's sign was exactly
        `value > segment_mean`. Real SHAP has no such relationship, so the two
        must disagree on a meaningful share of rows.
        """
        df = self._shap_frame()
        parsed = df["TopSHAPFeatures"].map(json.loads)

        agreements = total = 0
        for _, group in df.groupby("Segment"):
            means = group.select_dtypes("number").mean()
            for idx, feats in parsed.loc[group.index].items():
                for name, value in feats.items():
                    if name not in means.index:
                        continue
                    above = df.at[idx, name] > means[name]
                    total += 1
                    if above == (value > 0):
                        agreements += 1

        assert total > 0
        share = agreements / total
        assert share < 0.95, (
            f"{share:.1%} of SHAP signs match 'is above the segment mean' — that is "
            "the discredited approximation, not Shapley values."
        )

    def test_both_signs_occur(self):
        df = self._shap_frame()
        values = [v for feats in df["TopSHAPFeatures"].map(json.loads) for v in feats.values()]
        assert any(v > 0 for v in values), "no feature ever increases churn risk"
        assert any(v < 0 for v in values), "no feature ever decreases churn risk"
