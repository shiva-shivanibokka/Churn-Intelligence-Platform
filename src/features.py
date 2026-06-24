"""
Feature Engineering Pipeline
=============================
Mirrors the two-tier feature pattern used by Uber Michelangelo and Salesforce Einstein:
- Behavioral engagement signals (product depth, session frequency, recency decay)
- Composite scores (engagement index, stickiness, support risk)
- Spend trend signals (order value trajectory, discount sensitivity)

All features are engineered from raw behavioral columns — no demographics used
as primary signals, mirroring industry best practice for fairness and signal quality.
"""

import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

logger = logging.getLogger(__name__)

RAW_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "raw", "E Commerce Dataset.xlsx"
)
PROCESSED_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


EXPECTED_COLUMNS = [
    "CustomerID", "Churn", "Tenure", "PreferredLoginDevice", "CityTier",
    "WarehouseToHome", "PreferredPaymentMode", "Gender", "HourSpendOnApp",
    "NumberOfDeviceRegistered", "PreferedOrderCat", "SatisfactionScore",
    "MaritalStatus", "NumberOfAddress", "Complain", "OrderAmountHikeFromlastYear",
    "CouponUsed", "OrderCount", "DaySinceLastOrder", "CashbackAmount",
]

MAX_MISSING_RATE = 0.30


def validate_schema(df: pd.DataFrame) -> None:
    """
    Validate the raw dataset schema before feature engineering.
    Raises ValueError with a clear message if the data does not meet expectations.
    This prevents silent failures deep in the pipeline from a renamed column or
    wrong sheet.
    """
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Raw data is missing expected columns: {missing_cols}. "
            f"Check that the correct sheet ('E Comm') was loaded."
        )

    high_missing = {
        col: df[col].isna().mean()
        for col in EXPECTED_COLUMNS
        if df[col].isna().mean() > MAX_MISSING_RATE
    }
    if high_missing:
        logger.warning(
            "Columns with >%d%% missing values: %s",
            int(MAX_MISSING_RATE * 100), high_missing,
        )

    if len(df) < 100:
        raise ValueError(f"Dataset has only {len(df)} rows — expected at least 100.")

    logger.info("Schema validation passed: %d rows, %d columns.", len(df), len(df.columns))


def load_raw_data() -> pd.DataFrame:
    """Load the raw e-commerce dataset from xlsx."""
    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(
            f"Raw data file not found at '{RAW_PATH}'. "
            "Run: kaggle datasets download -d ankitverma2010/ecommerce-customer-churn-analysis-and-prediction "
            "-p data/raw --unzip"
        )
    try:
        df = pd.read_excel(RAW_PATH, sheet_name="E Comm")
    except Exception as exc:
        raise RuntimeError(
            f"Failed to read '{RAW_PATH}'. Ensure the file is a valid xlsx with sheet 'E Comm'."
        ) from exc
    validate_schema(df)
    return df


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values using median (numerical) strategy.
    Median is preferred over mean for skewed behavioral distributions.
    """
    numerical_cols = [
        "Tenure",
        "WarehouseToHome",
        "HourSpendOnApp",
        "OrderAmountHikeFromlastYear",
        "CouponUsed",
        "OrderCount",
        "DaySinceLastOrder",
    ]
    for col in numerical_cols:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Label-encode categorical columns.
    Ordinal encoding is used because tree-based models (XGBoost) handle
    label-encoded categoricals natively without inflating dimensionality.
    """
    cat_cols = [
        "PreferredLoginDevice",
        "PreferredPaymentMode",
        "Gender",
        "PreferedOrderCat",
        "MaritalStatus",
    ]
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build composite behavioral features.

    These mirror leading indicators documented in Uber/Spotify/Salesforce production systems:
    - Engagement depth (not just frequency, but HOW deeply customers use the product)
    - Recency decay (how recently did they last take a meaningful action)
    - Stickiness (cross-device usage signals commitment)
    - Support risk (complaint history is a leading churn indicator at Salesforce)
    - Spend trajectory (order value trend — declining spend predicts churn)
    - Discount sensitivity (heavy coupon use signals price-sensitivity, not loyalty)
    """

    # --- Engagement Score ---
    # Weighted composite: hours on app (depth) + order count (frequency)
    # Normalized to [0,1] range for interpretability
    df["EngagementScore"] = 0.5 * (
        df["HourSpendOnApp"] / df["HourSpendOnApp"].max()
    ) + 0.5 * (df["OrderCount"] / df["OrderCount"].max())

    # --- Recency Signal ---
    # Days since last order normalized. Higher = more lapsed.
    # Direct analog of Spotify's "days since last stream" leading indicator.
    _max_days = df["DaySinceLastOrder"].max()
    df["RecencySignal"] = df["DaySinceLastOrder"] / (_max_days if _max_days > 0 else 1)

    # --- Stickiness Index ---
    # Number of registered devices / number of addresses.
    # Multiple devices = deep ecosystem integration (hard to leave).
    # Multiple addresses = loyalty across life events (positive signal).
    df["StickinessIndex"] = (df["NumberOfDeviceRegistered"] + df["NumberOfAddress"]) / (
        df["NumberOfDeviceRegistered"].max() + df["NumberOfAddress"].max()
    )

    # --- Spend Trend ---
    # Year-over-year order amount change. Negative trend is a strong churn signal.
    # Normalized. At Salesforce, spend decline is a top-3 churn predictor.
    df["SpendTrend"] = df["OrderAmountHikeFromlastYear"] / (
        df["OrderAmountHikeFromlastYear"].max() + 1e-9
    )

    # --- Support Risk Score ---
    # Complain flag + satisfaction score inversion.
    # SatisfactionScore: 1=best, 5=worst. Invert so higher = more at risk.
    # This mirrors Salesforce's "support sentiment" health score component.
    df["SupportRiskScore"] = (
        df["Complain"] * 0.6
        + ((df["SatisfactionScore"] - 1) / 4) * 0.4  # normalized 0-1, higher = worse
    )

    # --- Discount Sensitivity ---
    # Heavy coupon usage relative to order count signals price-driven loyalty,
    # not brand loyalty — a leading churn indicator when discounts stop.
    df["DiscountSensitivity"] = df["CouponUsed"] / (df["OrderCount"] + 1e-9)

    # --- Tenure Stability ---
    # Long-tenured customers are inherently more stable.
    # Log-transform dampens the effect of extreme tenure values.
    df["TenureStability"] = np.log1p(df["Tenure"])

    # --- Warehouse Friction ---
    # Longer warehouse-to-home distance = more friction in the fulfillment experience.
    # Normalized to [0, 1].
    df["WarehouseFriction"] = df["WarehouseToHome"] / (
        df["WarehouseToHome"].max() + 1e-9
    )

    return df


def get_feature_sets() -> dict:
    """
    Returns the canonical feature sets used at each stage of the pipeline.
    Separating feature sets by stage mirrors how production feature stores
    (Uber Michelangelo, Airbnb Chronon, Stripe Shepherd) define feature groups.
    """
    return {
        # Features used for clustering (behavioral, no label leakage)
        "clustering": [
            "EngagementScore",
            "RecencySignal",
            "StickinessIndex",
            "SpendTrend",
            "SupportRiskScore",
            "DiscountSensitivity",
            "TenureStability",
            "WarehouseFriction",
            "CityTier",
            "HourSpendOnApp",
            "OrderCount",
            "NumberOfDeviceRegistered",
            "SatisfactionScore",
        ],
        # Full feature set for churn classification
        "churn_model": [
            "Tenure",
            "CityTier",
            "WarehouseToHome",
            "HourSpendOnApp",
            "NumberOfDeviceRegistered",
            "SatisfactionScore",
            "NumberOfAddress",
            "Complain",
            "OrderAmountHikeFromlastYear",
            "CouponUsed",
            "OrderCount",
            "DaySinceLastOrder",
            "CashbackAmount",
            "PreferredLoginDevice",
            "PreferredPaymentMode",
            "Gender",
            "PreferedOrderCat",
            "MaritalStatus",
            # Engineered features
            "EngagementScore",
            "RecencySignal",
            "StickinessIndex",
            "SpendTrend",
            "SupportRiskScore",
            "DiscountSensitivity",
            "TenureStability",
            "WarehouseFriction",
        ],
        # Minimal features exposed to uplift model (causal ML requires clean signal)
        "uplift_model": [
            "EngagementScore",
            "RecencySignal",
            "StickinessIndex",
            "SpendTrend",
            "SupportRiskScore",
            "TenureStability",
            "CityTier",
            "SatisfactionScore",
            "Complain",
        ],
    }


def build_pipeline(save: bool = True) -> pd.DataFrame:
    """
    Full feature engineering pipeline. Returns processed DataFrame.
    If save=True, writes processed data to disk (mirrors offline feature store materialization).
    """
    logger.info("Loading raw data...")
    df = load_raw_data()

    logger.info("Imputing missing values...")
    df = impute_missing(df)

    logger.info("Encoding categoricals...")
    df = encode_categoricals(df)

    logger.info("Engineering behavioral features...")
    df = engineer_features(df)

    if save:
        os.makedirs(PROCESSED_PATH, exist_ok=True)
        out_path = os.path.join(PROCESSED_PATH, "features.parquet")
        df.to_parquet(out_path, index=False)
        logger.info("Saved processed features to %s", out_path)

    logger.info("Pipeline complete. Shape: %s", df.shape)
    return df


if __name__ == "__main__":
    df = build_pipeline()
    print(
        df[
            [
                "EngagementScore",
                "RecencySignal",
                "StickinessIndex",
                "SpendTrend",
                "SupportRiskScore",
            ]
        ].describe()
    )
