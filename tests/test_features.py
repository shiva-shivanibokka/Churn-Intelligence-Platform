"""Tests for feature engineering pipeline."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from features import (
    encode_categoricals,
    engineer_features,
    impute_missing,
    validate_schema,
)


def make_raw_df(n: int = 120) -> pd.DataFrame:
    """Build a minimal valid raw DataFrame that matches the dataset schema."""
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "CustomerID": range(1, n + 1),
        "Churn": rng.integers(0, 2, n),
        "Tenure": rng.integers(1, 60, n).astype(float),
        "PreferredLoginDevice": rng.choice(["Mobile Phone", "Computer"], n),
        "CityTier": rng.integers(1, 4, n),
        "WarehouseToHome": rng.integers(5, 40, n).astype(float),
        "PreferredPaymentMode": rng.choice(["Debit Card", "Credit Card", "UPI"], n),
        "Gender": rng.choice(["Male", "Female"], n),
        "HourSpendOnApp": rng.uniform(0, 5, n),
        "NumberOfDeviceRegistered": rng.integers(1, 6, n),
        "PreferedOrderCat": rng.choice(["Laptop & Accessory", "Mobile", "Fashion"], n),
        "SatisfactionScore": rng.integers(1, 6, n),
        "MaritalStatus": rng.choice(["Single", "Married", "Divorced"], n),
        "NumberOfAddress": rng.integers(1, 10, n),
        "Complain": rng.integers(0, 2, n),
        "OrderAmountHikeFromlastYear": rng.uniform(10, 30, n),
        "CouponUsed": rng.integers(0, 10, n).astype(float),
        "OrderCount": rng.integers(1, 20, n).astype(float),
        "DaySinceLastOrder": rng.integers(0, 30, n).astype(float),
        "CashbackAmount": rng.uniform(50, 300, n),
    })


class TestValidateSchema:
    def test_passes_on_valid_df(self):
        df = make_raw_df()
        validate_schema(df)  # should not raise

    def test_raises_on_missing_column(self):
        df = make_raw_df().drop(columns=["Tenure"])
        with pytest.raises(ValueError, match="missing expected columns"):
            validate_schema(df)

    def test_raises_on_too_few_rows(self):
        df = make_raw_df(n=50)
        with pytest.raises(ValueError, match="only 50 rows"):
            validate_schema(df)

    def test_warns_on_high_missing_rate(self, caplog):
        df = make_raw_df()
        df["Tenure"] = np.nan  # 100% missing
        import logging
        with caplog.at_level(logging.WARNING, logger="features"):
            validate_schema(df)
        assert "Tenure" in caplog.text


class TestImputeMissing:
    def test_no_nans_after_imputation(self):
        df = make_raw_df()
        df.loc[0, "Tenure"] = np.nan
        df.loc[1, "DaySinceLastOrder"] = np.nan
        result = impute_missing(df)
        impute_cols = ["Tenure", "WarehouseToHome", "HourSpendOnApp",
                       "OrderAmountHikeFromlastYear", "CouponUsed",
                       "OrderCount", "DaySinceLastOrder"]
        assert result[impute_cols].isna().sum().sum() == 0

    def test_median_used_for_imputation(self):
        df = make_raw_df(n=100)
        median_tenure = df["Tenure"].median()
        df.loc[0, "Tenure"] = np.nan
        result = impute_missing(df)
        assert result.loc[0, "Tenure"] == median_tenure


class TestEncodeCategoricals:
    def test_categorical_cols_are_numeric(self):
        df = make_raw_df()
        result = encode_categoricals(df)
        cat_cols = ["PreferredLoginDevice", "PreferredPaymentMode",
                    "Gender", "PreferedOrderCat", "MaritalStatus"]
        for col in cat_cols:
            assert pd.api.types.is_numeric_dtype(result[col]), f"{col} should be numeric"


class TestEngineerFeatures:
    def test_all_engineered_columns_present(self):
        df = make_raw_df()
        df = impute_missing(df)
        df = encode_categoricals(df)
        result = engineer_features(df)
        expected = [
            "EngagementScore", "RecencySignal", "StickinessIndex",
            "SpendTrend", "SupportRiskScore", "DiscountSensitivity",
            "TenureStability", "WarehouseFriction",
        ]
        for col in expected:
            assert col in result.columns, f"Missing engineered feature: {col}"

    def test_engagement_score_in_range(self):
        df = make_raw_df(n=100)
        df = impute_missing(df)
        df = encode_categoricals(df)
        result = engineer_features(df)
        assert result["EngagementScore"].between(0, 1).all()

    def test_recency_signal_in_range(self):
        df = make_raw_df(n=100)
        df = impute_missing(df)
        df = encode_categoricals(df)
        result = engineer_features(df)
        assert result["RecencySignal"].between(0, 1).all()

    def test_support_risk_score_in_range(self):
        df = make_raw_df(n=100)
        df = impute_missing(df)
        df = encode_categoricals(df)
        result = engineer_features(df)
        assert result["SupportRiskScore"].between(0, 1).all()
