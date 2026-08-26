"""
Tests for the uplift sign convention.

This file exists because the sign was wrong in production and nothing caught
it. `compute_uplift_scores_causalml` used CausalML's `predict()` output
directly, and CausalML returns the treatment effect on the *outcome*:

    predict(X) = mu_1(x) - mu_0(x) = P(churn | treated) - P(churn | control)

The outcome here is churn, which we want to go down, so a large positive
CausalML score marks the customer intervention is expected to *drive away*.
`classify_customer_type` read that same number as "responds well to
intervention", so every customer promoted to "Persuadable" was really the
model's strongest Sleeping Dog — the one group the whole four-quadrant framing
exists to keep you away from.

The wrong sign produced nothing that looked wrong: a ranked list, plausible ROI
figures, a full dashboard, 798 names for the retention agent to write plans
about. The only way to catch it was to check the score against observed
outcomes, which is what `validate_uplift_direction` does and what these tests
pin down.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from uplift_model import (  # noqa: E402
    classify_customer_type,
    estimate_intervention_roi,
    simulate_treatment,
    validate_uplift_direction,
)


def _synthetic(n: int = 2000, inverted: bool = False) -> pd.DataFrame:
    """
    A population where the true effect is known by construction.

    Half are treated. Each customer has a latent `responsiveness`: high means
    the intervention helps them a lot. Churn is generated so that treated,
    responsive customers churn less. `UpliftScore` is then written in the
    project's convention (positive = intervention reduces churn), or negated
    when `inverted=True` to reproduce the shipped bug.
    """
    rng = np.random.default_rng(0)
    responsiveness = rng.uniform(-0.3, 0.3, n)
    treated = rng.integers(0, 2, n)

    base_churn = 0.4
    p_churn = base_churn - treated * responsiveness
    churn = (rng.uniform(0, 1, n) < p_churn).astype(int)

    scores = -responsiveness if inverted else responsiveness
    return pd.DataFrame(
        {
            "Treatment": treated,
            "Churn": churn,
            "UpliftScore": scores,
            "ChurnProbability": p_churn,
            "CustomerType": [
                classify_customer_type(u, c) for u, c in zip(scores, p_churn, strict=True)
            ],
        }
    )


class TestValidateUpliftDirection:
    def test_correct_sign_passes(self):
        result = validate_uplift_direction(_synthetic())
        assert result["direction_ok"] is True
        assert result["churn_top_uplift_decile"] < result["churn_bottom_uplift_decile"]

    def test_inverted_sign_is_caught(self):
        """The exact shipped bug, and the check that would have flagged it."""
        result = validate_uplift_direction(_synthetic(inverted=True))
        assert result["direction_ok"] is False
        assert result["churn_top_uplift_decile"] > result["churn_bottom_uplift_decile"]

    def test_too_little_data_is_unknown_not_a_pass(self):
        """
        An inconclusive check must not read as a passing one. Returning True
        here would mean a tiny run silently certifies its own sign.
        """
        tiny = _synthetic(n=20)
        assert validate_uplift_direction(tiny)["direction_ok"] is None

    def test_committed_artifacts_have_the_right_sign(self):
        """
        The real check, against the data that is actually shipped. Skips rather
        than fails when the pipeline has not been run, so a fresh clone is not
        blocked, but a wrong-signed artifact cannot be committed unnoticed.
        """
        path = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "uplift.parquet")
        if not os.path.exists(path):
            pytest.skip("uplift.parquet not present — run python src/pipeline.py first")

        df = pd.read_parquet(path)
        if "Treatment" not in df.columns:
            df["Treatment"] = simulate_treatment(df)

        result = validate_uplift_direction(df)
        assert result["direction_ok"] is True, (
            "The committed UpliftScore points the wrong way: among treated "
            f"customers the top decile churns at {result['churn_top_uplift_decile']} "
            f"against the bottom decile's {result['churn_bottom_uplift_decile']}. "
            "Everything labelled Persuadable is a Sleeping Dog."
        )


class TestClassifyCustomerType:
    """The four quadrants, including the boundaries the thresholds sit on."""

    def test_persuadable(self):
        assert classify_customer_type(uplift_score=0.10, churn_prob=0.50) == "Persuadable"

    def test_sure_thing(self):
        assert classify_customer_type(uplift_score=0.10, churn_prob=0.10) == "Sure Thing"

    def test_lost_cause(self):
        assert classify_customer_type(uplift_score=-0.10, churn_prob=0.50) == "Lost Cause"

    def test_sleeping_dog(self):
        assert classify_customer_type(uplift_score=-0.10, churn_prob=0.10) == "Sleeping Dog"

    def test_boundary_churn_threshold(self):
        # 0.30 uses >=, so 0.29 is not high churn.
        assert classify_customer_type(uplift_score=0.10, churn_prob=0.29) == "Sure Thing"

    def test_boundary_uplift_threshold(self):
        # Exactly at 0.05 counts as positive uplift.
        assert classify_customer_type(uplift_score=0.05, churn_prob=0.50) == "Persuadable"

    def test_custom_thresholds(self):
        assert classify_customer_type(
            uplift_score=0.03, churn_prob=0.50,
            uplift_threshold=0.02, churn_threshold=0.40,
        ) == "Persuadable"

    def test_a_negative_score_can_never_be_persuadable(self):
        """
        The property the sign bug violated end to end: a customer the model
        expects to be harmed by contact must never reach the target list,
        whatever their churn probability.
        """
        for churn in (0.05, 0.31, 0.7, 0.99):
            assert classify_customer_type(-0.2, churn) in ("Lost Cause", "Sleeping Dog")


class TestInterventionRoi:
    def test_roi_follows_uplift_sign(self):
        """
        ROI is `uplift x CLV - cost`, so an inverted uplift does not merely
        mislabel customers — it inverts the ranking used to spend the budget.
        """
        df = pd.DataFrame(
            {
                "UpliftScore": [0.20, -0.20],
                "ChurnProbability": [0.6, 0.6],
                "CustomerType": ["Persuadable", "Lost Cause"],
            }
        )
        out = estimate_intervention_roi(df, avg_clv=500.0, intervention_cost=15.0)
        assert out["NetROI"].iloc[0] == pytest.approx(0.20 * 500 - 15)
        assert out["NetROI"].iloc[1] == pytest.approx(-0.20 * 500 - 15)
        assert bool(out["ROIPositive"].iloc[0]) is True
        assert bool(out["ROIPositive"].iloc[1]) is False

    def test_priority_is_only_assigned_to_persuadables(self):
        df = pd.DataFrame(
            {
                "UpliftScore": [0.20, 0.18, -0.20],
                "ChurnProbability": [0.6, 0.6, 0.6],
                "CustomerType": ["Persuadable", "Persuadable", "Lost Cause"],
            }
        )
        out = estimate_intervention_roi(df)
        assert out["InterventionPriority"].notna().sum() == 2
        assert pd.isna(out["InterventionPriority"].iloc[2])
        # Highest ROI ranks first.
        assert out["InterventionPriority"].iloc[0] == 1.0
