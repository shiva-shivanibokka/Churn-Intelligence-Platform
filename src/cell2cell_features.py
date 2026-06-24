"""
Cell2Cell Subscription Churn Feature Engineering
==================================================
Loads the Cell2Cell wireless telecom churn dataset (~71K subscribers) and
engineers a feature set compatible with the segmentation → churn → uplift pipeline.

Dataset: search "cell2cell churn" on Kaggle, or download from Duke Fuqua.
Kaggle:  kaggle datasets download -d jpacse/cellc2cell2000 -p data/raw/cell2cell --unzip

Why Cell2Cell for subscription churn:
- Pre-labeled Churn column (~30% churn rate — no engineering needed)
- 71K rows — large enough to justify XGBoost + LightGBM + CatBoost ensemble
- Subscription mechanics mirror Netflix, Spotify, SaaS: monthly billing,
  usage patterns, customer service contacts, retention interventions

Leaky columns excluded (they were recorded AFTER churn was known):
- RetentionCalls, RetentionOffersAccepted, MadeCallToRetentionTeam
"""

import logging
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

CELL2CELL_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "cell2cell")
PROCESSED_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

# Columns that would cause data leakage — recorded post-churn-decision
LEAKY_COLS = {"RetentionCalls", "RetentionOffersAccepted", "MadeCallToRetentionTeam"}

# Map yes/no columns to 1/0
YES_NO_COLS = [
    "ChildrenInHH", "HandsetRefurbished", "HandsetWebCapable", "TruckOwner",
    "RVOwner", "BuysViaMailOrder", "RespondsToMailOffers", "OptOutMailings",
    "NonUSTravel", "OwnsComputer", "HasCreditCard", "OwnsMotorcycle",
]

# CreditRating ordinal map
CREDIT_RATING_MAP = {
    "1-Highest": 5, "2-High": 4, "3-Good": 3, "4-Medium": 2,
    "5-Low": 1, "6-VeryLow": 0, "7-Lowest": 0,
}


def _find_csv(path: str) -> str:
    """Find the Cell2Cell CSV — handles different filenames from different sources."""
    candidates = [
        "cell2celltrain.csv", "Cell2CellTrain.csv", "CELL2CELLTRAIN.csv",
        "cell2cell_train.csv", "train.csv", "TRAIN.csv",
        "cell2cell.csv", "Cell2Cell.csv", "cell2cell_data.csv",
    ]
    for fname in candidates:
        fpath = os.path.join(path, fname)
        if os.path.exists(fpath):
            return fpath
    # Fall back to first CSV that isn't the holdout (holdout has no Churn label)
    for f in sorted(os.listdir(path)):
        if f.lower().endswith(".csv") and "holdout" not in f.lower():
            return os.path.join(path, f)
    raise FileNotFoundError(
        f"No CSV found in {path}.\n"
        "Download with: kaggle datasets download -d jpacse/cellc2cell2000 "
        "-p data/raw/cell2cell --unzip"
    )


def load_cell2cell() -> pd.DataFrame:
    if not os.path.exists(CELL2CELL_PATH):
        raise FileNotFoundError(
            f"Directory not found: {CELL2CELL_PATH}\n"
            "Download with: kaggle datasets download -d jpacse/cellc2cell2000 "
            "-p data/raw/cell2cell --unzip"
        )
    fpath = _find_csv(CELL2CELL_PATH)
    df = pd.read_csv(fpath)
    logger.info("Loaded Cell2Cell from %s: %d rows, %d columns", fpath, len(df), len(df.columns))
    return df


def clean_cell2cell(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Standardise column names (some versions use different capitalisation)
    df.columns = [c.strip() for c in df.columns]

    # Drop leaky columns if present
    drop_cols = [c for c in df.columns if c in LEAKY_COLS]
    if drop_cols:
        logger.info("Dropping leaky columns: %s", drop_cols)
        df = df.drop(columns=drop_cols)

    # Churn column — normalise to integer 0/1
    churn_col = next((c for c in df.columns if c.lower() == "churn"), None)
    if churn_col is None:
        raise ValueError("No 'Churn' column found. Check the dataset.")
    df["Churn"] = df[churn_col].map(
        lambda x: 1 if str(x).strip().lower() in ("1", "yes", "true") else 0
    )

    # Yes/No → 1/0
    for col in YES_NO_COLS:
        if col in df.columns:
            df[col] = df[col].map(
                lambda x: 1 if str(x).strip().lower() in ("1", "yes", "true") else 0
            )

    # CreditRating → ordinal integer
    if "CreditRating" in df.columns:
        df["CreditRating"] = df["CreditRating"].map(CREDIT_RATING_MAP).fillna(2)

    # Homeownership → binary (Known Owner = 1, else 0)
    if "Homeownership" in df.columns:
        df["Homeownership"] = (df["Homeownership"].astype(str).str.lower() == "known homeowner").astype(int)

    # IncomeGroup → numeric (strip if it's a string like "1" or "IncomeGroup1")
    if "IncomeGroup" in df.columns:
        df["IncomeGroup"] = pd.to_numeric(
            df["IncomeGroup"].astype(str).str.extract(r"(\d+)")[0], errors="coerce"
        ).fillna(3)

    # Label-encode remaining object columns
    for col in df.select_dtypes(include="object").columns:
        if col not in ("CustomerID", "Churn"):
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str).fillna("Unknown"))

    # CustomerID — use index if not present
    if "CustomerID" not in df.columns and "Customer_ID" not in df.columns:
        df["CustomerID"] = df.index.astype(str)
    elif "Customer_ID" in df.columns:
        df = df.rename(columns={"Customer_ID": "CustomerID"})

    df["CustomerID"] = df["CustomerID"].astype(str)

    # Drop rows with no Churn label
    df = df[df["Churn"].notna()].copy()

    logger.info(
        "Cleaned Cell2Cell: %d rows | Churn rate: %.1f%%",
        len(df), df["Churn"].mean() * 100
    )
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map Cell2Cell columns to pipeline-compatible feature names and
    engineer the same behavioral composites used in the E-Commerce and Olist pipelines.

    Subscription → E-Commerce mapping:
    - MonthsInService      → Tenure (months as subscriber)
    - MonthlyRevenue       → AvgOrderValue (monthly spend)
    - MonthlyMinutes       → HourSpendOnApp (usage intensity)
    - CustomerCareCalls    → Complain proxy
    - CreditRating         → SatisfactionScore proxy
    - Handsets             → NumberOfDeviceRegistered
    - PercChangeRevenues   → OrderAmountHikeFromlastYear
    - RespondsToMailOffers → CouponUsed (responds to offers)
    - RoamingCalls         → WarehouseToHome (extra-service usage)
    - UniqueSubs           → NumberOfAddress
    - IncomeGroup          → CityTier
    - Occupation           → PreferedOrderCat
    """
    df = df.copy()

    # ── Core feature mapping ─────────────────────────────────────────────────
    tenure_col = "MonthsInService" if "MonthsInService" in df.columns else None
    df["Tenure"] = df[tenure_col].clip(lower=1) if tenure_col else 12

    df["AvgOrderValue"] = df.get("MonthlyRevenue", pd.Series(50, index=df.index)).fillna(50).clip(lower=0)
    df["OrderCount"] = df["Tenure"]  # billing cycles ≈ months subscribed

    df["HourSpendOnApp"] = (
        df.get("MonthlyMinutes", pd.Series(0, index=df.index)).fillna(0) / 100
    ).clip(0, 10)

    df["NumberOfDeviceRegistered"] = df.get("Handsets", pd.Series(1, index=df.index)).fillna(1).clip(1, 5)

    df["SatisfactionScore"] = df.get("CreditRating", pd.Series(3, index=df.index)).fillna(3).clip(1, 5).round(0).astype(int)

    df["Complain"] = (df.get("CustomerCareCalls", pd.Series(0, index=df.index)).fillna(0) > 3).astype(int)

    pct_change = df.get("PercChangeRevenues", pd.Series(0, index=df.index)).fillna(0)
    df["OrderAmountHikeFromlastYear"] = pct_change.clip(lower=0)

    df["DaySinceLastOrder"] = df.get("CurrentEquipmentDays", pd.Series(180, index=df.index)).fillna(180).clip(lower=0)

    df["CashbackAmount"] = df["AvgOrderValue"] * df.get("RespondsToMailOffers", pd.Series(0, index=df.index)).fillna(0)

    df["CouponUsed"] = df.get("RespondsToMailOffers", pd.Series(0, index=df.index)).fillna(0).astype(int)

    df["WarehouseToHome"] = df.get("RoamingCalls", pd.Series(0, index=df.index)).fillna(0).clip(0, 50)

    df["NumberOfAddress"] = df.get("UniqueSubs", pd.Series(1, index=df.index)).fillna(1).clip(1, 5)

    df["CityTier"] = df.get("IncomeGroup", pd.Series(3, index=df.index)).fillna(3).clip(1, 3).astype(int)

    df["PreferredPaymentMode"] = df.get("Homeownership", pd.Series(0, index=df.index)).fillna(0).astype(int)

    df["PreferedOrderCat"] = df.get("Occupation", pd.Series(0, index=df.index)).fillna(0)
    if df["PreferedOrderCat"].dtype == object:
        le = LabelEncoder()
        df["PreferedOrderCat"] = le.fit_transform(df["PreferedOrderCat"].astype(str))

    df["MaritalStatus"] = df.get("MaritalStatus", pd.Series(0, index=df.index)).fillna(0)
    if df["MaritalStatus"].dtype == object:
        le = LabelEncoder()
        df["MaritalStatus"] = le.fit_transform(df["MaritalStatus"].astype(str))

    df["Gender"] = 0  # not in Cell2Cell
    df["PreferredLoginDevice"] = df.get("HandsetWebCapable", pd.Series(0, index=df.index)).fillna(0).astype(int)

    # ── Behavioral composites (same formulas as features.py) ─────────────────
    df["EngagementScore"] = (
        0.5 * (df["HourSpendOnApp"] / max(df["HourSpendOnApp"].max(), 1))
        + 0.5 * (df["OrderCount"] / max(df["OrderCount"].max(), 1))
    )

    _max_days = df["DaySinceLastOrder"].max()
    df["RecencySignal"] = df["DaySinceLastOrder"] / (_max_days if _max_days > 0 else 1)

    df["StickinessIndex"] = (df["NumberOfDeviceRegistered"] + df["NumberOfAddress"]) / (
        df["NumberOfDeviceRegistered"].max() + df["NumberOfAddress"].max()
    )

    df["SpendTrend"] = df["OrderAmountHikeFromlastYear"] / (
        df["OrderAmountHikeFromlastYear"].max() + 1e-9
    )

    df["SupportRiskScore"] = (
        df["Complain"] * 0.6
        + ((df["SatisfactionScore"] - 1) / 4) * 0.4
    )

    df["DiscountSensitivity"] = df["CouponUsed"] / (df["OrderCount"] + 1e-9)

    df["TenureStability"] = np.log1p(df["Tenure"])

    df["WarehouseFriction"] = df["WarehouseToHome"] / (df["WarehouseToHome"].max() + 1e-9)

    return df


def get_cell2cell_feature_sets() -> dict:
    return {
        "clustering": [
            "EngagementScore", "RecencySignal", "StickinessIndex", "SpendTrend",
            "SupportRiskScore", "DiscountSensitivity", "TenureStability",
            "WarehouseFriction", "CityTier", "HourSpendOnApp", "OrderCount",
            "NumberOfDeviceRegistered", "SatisfactionScore",
        ],
        "churn_model": [
            "Tenure", "CityTier", "WarehouseToHome", "HourSpendOnApp",
            "NumberOfDeviceRegistered", "SatisfactionScore", "NumberOfAddress",
            "Complain", "OrderAmountHikeFromlastYear", "CouponUsed", "OrderCount",
            "DaySinceLastOrder", "CashbackAmount", "PreferredPaymentMode",
            "PreferedOrderCat", "EngagementScore", "RecencySignal", "StickinessIndex",
            "SpendTrend", "SupportRiskScore", "DiscountSensitivity",
            "TenureStability", "WarehouseFriction",
        ],
        "uplift_model": [
            "EngagementScore", "RecencySignal", "StickinessIndex", "SpendTrend",
            "SupportRiskScore", "TenureStability", "CityTier",
            "SatisfactionScore", "Complain",
        ],
    }


def build_cell2cell_pipeline(save: bool = True) -> pd.DataFrame:
    logger.info("Loading Cell2Cell dataset from %s", CELL2CELL_PATH)
    raw = load_cell2cell()

    logger.info("Cleaning and encoding...")
    df = clean_cell2cell(raw)

    logger.info("Engineering features...")
    df = engineer_features(df)

    # Fill remaining NaNs
    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    if save:
        os.makedirs(PROCESSED_PATH, exist_ok=True)
        out = os.path.join(PROCESSED_PATH, "features.parquet")
        df.to_parquet(out, index=False)
        logger.info("Saved to %s", out)

    logger.info(
        "Cell2Cell pipeline complete. Shape: %s | Churn rate: %.1f%%",
        df.shape, df["Churn"].mean() * 100
    )
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    df = build_cell2cell_pipeline(save=True)
    print("\nChurn distribution:")
    print(df["Churn"].value_counts())
    print("\nFeature summary:")
    print(df[["Tenure", "AvgOrderValue", "HourSpendOnApp", "SatisfactionScore",
              "Complain", "EngagementScore"]].describe().round(3))
