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
import os

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from composite_features import add_composite_features, fit_composite_norms

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


def engineer_features(df: pd.DataFrame, norms: dict | None = None) -> pd.DataFrame:
    """
    Build the eight composite behavioural features.

    Delegates to src/composite_features.py so that all three dataset modules
    share one implementation. Pass `norms` (from `fit_composite_norms` on the
    training frame, persisted to models/feature_norms.json) whenever scoring
    rows that are not the training population — without it the normalisers are
    refitted on whatever was handed in, which silently pins five of the eight
    features to 1.0 for a single-row request.
    """
    if norms is None:
        norms = fit_composite_norms(df)
    return add_composite_features(df, norms)



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
