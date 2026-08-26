"""
Tests for the composite features, and specifically for their normalisers.

Five of the eight composites divide by a column maximum. When those maxima were
recomputed inside every call, scoring a single customer through the API divided
each feature by itself, so `EngagementScore`, `RecencySignal`,
`StickinessIndex`, `SpendTrend` and `WarehouseFriction` were all exactly 1.0 for
every request that endpoint ever served.

Nothing failed. The model accepted the frame, returned a probability, and the
endpoint answered 200 — it just answered almost the same thing every time,
because five of its inputs were constants. The property below is the one that
distinguishes the two situations, and it is cheap: a customer must score the
same alone as they do inside the training batch.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from composite_features import (  # noqa: E402
    COMPOSITE_FEATURES,
    NORM_COLUMNS,
    add_composite_features,
    fit_composite_norms,
)


@pytest.fixture
def population() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "HourSpendOnApp": [1.0, 3.0, 5.0, 2.5],
            "OrderCount": [2.0, 10.0, 40.0, 7.0],
            "DaySinceLastOrder": [1.0, 15.0, 60.0, 30.0],
            "NumberOfDeviceRegistered": [1, 3, 5, 2],
            "NumberOfAddress": [1, 2, 4, 3],
            "OrderAmountHikeFromlastYear": [5.0, 15.0, 30.0, 12.0],
            "WarehouseToHome": [5.0, 15.0, 40.0, 22.0],
            "Complain": [0, 1, 0, 1],
            "SatisfactionScore": [5, 1, 3, 2],
            "CouponUsed": [0.0, 4.0, 1.0, 2.0],
            "Tenure": [2.0, 24.0, 60.0, 10.0],
        }
    )


class TestServingConsistency:
    def test_single_row_matches_batch(self, population):
        """The regression: one row scored alone must equal that row in the batch."""
        norms = fit_composite_norms(population)
        batch = add_composite_features(population.copy(), norms)

        for i in range(len(population)):
            single = add_composite_features(population.iloc[[i]].copy(), norms)
            for feat in COMPOSITE_FEATURES:
                assert single[feat].iloc[0] == pytest.approx(
                    batch[feat].iloc[i], abs=0, rel=1e-12
                ), f"row {i}, feature {feat}"

    def test_refitting_on_one_row_is_what_went_wrong(self, population):
        """
        Demonstrates the old behaviour so the fix has something to be measured
        against: refitting per call pins the batch-normalised features to 1.0.
        """
        refit_on_one_row = add_composite_features(
            population.iloc[[0]].copy(), fit_composite_norms(population.iloc[[0]])
        )
        for feat in ("EngagementScore", "RecencySignal", "StickinessIndex",
                     "SpendTrend", "WarehouseFriction"):
            assert refit_on_one_row[feat].iloc[0] == pytest.approx(1.0)

        correct = add_composite_features(
            population.iloc[[0]].copy(), fit_composite_norms(population)
        )
        assert correct["EngagementScore"].iloc[0] < 0.3
        assert correct["RecencySignal"].iloc[0] < 0.1

    def test_row_local_features_are_unaffected_by_norms(self, population):
        """
        These three are ratios and transforms of one row, so they were always
        correct at serving time. Pinning that down stops a future refactor from
        making them depend on the batch.
        """
        wide = fit_composite_norms(population)
        narrow = {k: v * 10 for k, v in wide.items()}
        a = add_composite_features(population.copy(), wide)
        b = add_composite_features(population.copy(), narrow)
        for feat in ("SupportRiskScore", "DiscountSensitivity", "TenureStability"):
            assert np.allclose(a[feat], b[feat])


class TestRobustness:
    def test_all_eight_features_are_produced(self, population):
        out = add_composite_features(population.copy(), fit_composite_norms(population))
        for feat in COMPOSITE_FEATURES:
            assert feat in out.columns

    def test_zero_column_does_not_produce_nan_or_inf(self, population):
        """A dataset where nobody has any distance must not yield NaN."""
        population["WarehouseToHome"] = 0.0
        out = add_composite_features(population.copy(), fit_composite_norms(population))
        assert np.isfinite(out["WarehouseFriction"]).all()

    def test_missing_norms_raise_rather_than_silently_defaulting(self, population):
        """
        A norms dict from an older artifact must fail loudly. Defaulting the
        missing entry to zero would divide by the epsilon floor and produce
        enormous feature values that still look like numbers.
        """
        incomplete = fit_composite_norms(population)
        del incomplete["WarehouseToHome"]
        with pytest.raises(ValueError, match="WarehouseToHome"):
            add_composite_features(population.copy(), incomplete)

    def test_norms_cover_every_column_the_transform_divides_by(self, population):
        assert set(fit_composite_norms(population)) == set(NORM_COLUMNS)


class TestCommittedNorms:
    def test_saved_norms_reproduce_the_committed_features(self):
        """
        The two-hop check: the norms on disk must be the ones the committed
        parquet was built with. If they drift, everything scored through the API
        silently disagrees with everything on the dashboard.
        """
        root = os.path.join(os.path.dirname(__file__), "..")
        norms_path = os.path.join(root, "models", "feature_norms.json")
        parquet_path = os.path.join(root, "data", "processed", "uplift.parquet")
        if not (os.path.exists(norms_path) and os.path.exists(parquet_path)):
            pytest.skip("artifacts not present — run python src/pipeline.py first")

        import json

        with open(norms_path, encoding="utf-8") as fh:
            norms = json.load(fh)

        df = pd.read_parquet(parquet_path)
        replayed = add_composite_features(df.copy(), norms)
        for feat in COMPOSITE_FEATURES:
            assert np.allclose(replayed[feat], df[feat], rtol=1e-9, atol=1e-12), (
                f"models/feature_norms.json does not reproduce '{feat}' in the "
                "committed parquet — the saved constants and the shipped data "
                "were produced by different runs."
            )
