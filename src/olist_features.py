"""
Olist E-Commerce Feature Engineering
======================================
Loads the Brazilian Olist public dataset (~100K orders, 96K customers) and
engineers a feature set analogous to the existing E-Commerce churn dataset.

Dataset: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
Download: kaggle datasets download -d olistbr/brazilian-ecommerce -p data/raw/olist --unzip

Key design decisions:
- Churn label: customer ordered before CHURN_CUTOFF_DATE but not after OBSERVATION_START.
  This creates a 6-month holdout window. 97% of Olist customers order once, so
  churn rate is naturally high (~85%). CatBoost's class_weights handles this.
- Features: mapped as closely as possible to the existing E-Commerce feature set so
  segmentation, churn, and uplift pipelines run unchanged.
- Columns not available in Olist (HourSpendOnApp, Gender, etc.) are approximated
  from available signals or set to sensible defaults documented below.
"""

import logging
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

OLIST_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "olist")
PROCESSED_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

# 6-month observation window near the end of the dataset (Sep 2016 – Oct 2018)
CHURN_CUTOFF_DATE   = pd.Timestamp("2018-06-01")   # feature engineering cutoff
OBSERVATION_START   = pd.Timestamp("2018-06-01")   # if NO order after this → churned
OBSERVATION_END     = pd.Timestamp("2018-10-01")   # dataset end

# Brazilian state → City Tier (1=major metro, 2=secondary, 3=other)
STATE_TIER = {
    "SP": 1, "RJ": 1,
    "MG": 2, "RS": 2, "PR": 2, "BA": 2,
    "SC": 3, "PE": 3, "CE": 3, "GO": 3, "ES": 3, "AM": 3,
    "MT": 3, "MS": 3, "RN": 3, "PB": 3, "AL": 3, "SE": 3,
    "PA": 3, "PI": 3, "MA": 3, "TO": 3, "AP": 3, "AC": 3, "RO": 3, "RR": 3, "DF": 2,
}

PAYMENT_MODE_MAP = {
    "credit_card": 0, "boleto": 1, "voucher": 2, "debit_card": 3,
}


def load_olist_csvs() -> dict[str, pd.DataFrame]:
    """Load the 5 Olist CSVs needed for feature engineering."""
    files = {
        "customers": "olist_customers_dataset.csv",
        "orders":    "olist_orders_dataset.csv",
        "items":     "olist_order_items_dataset.csv",
        "payments":  "olist_order_payments_dataset.csv",
        "reviews":   "olist_order_reviews_dataset.csv",
        "products":  "olist_products_dataset.csv",
    }
    dfs = {}
    for key, fname in files.items():
        fpath = os.path.join(OLIST_PATH, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(
                f"Missing Olist file: {fpath}\n"
                "Download with: kaggle datasets download -d olistbr/brazilian-ecommerce "
                "-p data/raw/olist --unzip"
            )
        dfs[key] = pd.read_csv(fpath)
        logger.info("Loaded %s: %d rows", fname, len(dfs[key]))
    return dfs


def build_order_features(dfs: dict) -> pd.DataFrame:
    """
    Join Olist tables and compute per-order features.
    Returns one row per (customer_unique_id, order_id).
    """
    orders = dfs["orders"].copy()
    orders["order_purchase_timestamp"] = pd.to_datetime(orders["order_purchase_timestamp"])

    # Merge customer info to get customer_unique_id and state
    orders = orders.merge(dfs["customers"][["customer_id", "customer_unique_id", "customer_state"]], on="customer_id", how="left")

    # Merge payment info (aggregate by order)
    pay = dfs["payments"].groupby("order_id").agg(
        total_payment=("payment_value", "sum"),
        avg_installments=("payment_installments", "mean"),
        preferred_payment=("payment_type", lambda x: x.mode()[0] if len(x) > 0 else "credit_card"),
    ).reset_index()
    orders = orders.merge(pay, on="order_id", how="left")

    # Merge item info (aggregate by order)
    items = dfs["items"].groupby("order_id").agg(
        item_count=("order_item_id", "count"),
        total_price=("price", "sum"),
        avg_freight=("freight_value", "mean"),
        distinct_sellers=("seller_id", "nunique"),
    ).reset_index()
    orders = orders.merge(items, on="order_id", how="left")

    # Merge product categories
    if "product_category_name" in dfs["products"].columns:
        prod_cat = dfs["items"].merge(dfs["products"][["product_id", "product_category_name"]], on="product_id", how="left")
        dominant_cat = prod_cat.groupby("order_id")["product_category_name"].agg(
            lambda x: x.dropna().mode()[0] if x.dropna().shape[0] > 0 else "other"
        ).reset_index().rename(columns={"product_category_name": "dominant_category"})
        orders = orders.merge(dominant_cat, on="order_id", how="left")
    else:
        orders["dominant_category"] = "other"

    # Merge reviews (most recent per order)
    reviews = dfs["reviews"].copy()
    # Some orders have multiple reviews — take the most recent
    reviews = reviews.sort_values("review_answer_timestamp", na_position="last")\
                     .drop_duplicates("order_id", keep="last")[["order_id", "review_score"]]
    orders = orders.merge(reviews, on="order_id", how="left")
    orders["review_score"] = orders["review_score"].fillna(3.0)  # median imputation

    # Only keep delivered orders (excludes cancelled/unavailable)
    orders = orders[orders["order_status"] == "delivered"].copy()
    logger.info("Delivered orders: %d", len(orders))

    return orders


def engineer_customer_features(orders: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate from order-level to customer-level, engineer churn label and features.
    """
    # Only use orders before the feature cutoff
    hist = orders[orders["order_purchase_timestamp"] < CHURN_CUTOFF_DATE].copy()

    if len(hist) == 0:
        raise ValueError("No orders found before CHURN_CUTOFF_DATE. Check your data path.")

    # ── Churn label ──────────────────────────────────────────────────────────
    # Future orders: placed in the observation window (cutoff to dataset end)
    future = orders[
        (orders["order_purchase_timestamp"] >= OBSERVATION_START) &
        (orders["order_purchase_timestamp"] < OBSERVATION_END)
    ][["customer_unique_id"]].drop_duplicates()
    future["_ordered_again"] = 1

    # Start from customers who had at least one historical order
    customers_hist = hist.groupby("customer_unique_id")

    # ── Per-customer aggregations ────────────────────────────────────────────
    agg = customers_hist.agg(
        OrderCount                   = ("order_id", "nunique"),
        first_order_date             = ("order_purchase_timestamp", "min"),
        last_order_date              = ("order_purchase_timestamp", "max"),
        AvgOrderValue                = ("total_payment", "mean"),
        TotalSpend                   = ("total_payment", "sum"),
        AvgFreightValue              = ("avg_freight", "mean"),
        AvgReviewScore               = ("review_score", "mean"),
        LowReviewCount               = ("review_score", lambda x: (x <= 2).sum()),
        TotalItemCount               = ("item_count", "sum"),
        AvgInstallments              = ("avg_installments", "mean"),
        DistinctSellers              = ("distinct_sellers", "sum"),
        CustomerState                = ("customer_state", "first"),
        PreferredPaymentMode         = ("preferred_payment", lambda x: x.mode()[0] if len(x) > 0 else "credit_card"),
        PreferedOrderCat             = ("dominant_category", lambda x: x.mode()[0] if x.dropna().shape[0] > 0 else "other"),
    ).reset_index()

    # Compute derived columns
    agg["Tenure"] = (CHURN_CUTOFF_DATE - agg["first_order_date"]).dt.days.clip(lower=1)
    agg["DaySinceLastOrder"] = (CHURN_CUTOFF_DATE - agg["last_order_date"]).dt.days.clip(lower=0)

    # YoY spend change: compare spend in 2nd half of history vs 1st half
    mid_date = agg["first_order_date"] + (CHURN_CUTOFF_DATE - agg["first_order_date"]) / 2
    # Simplified: approximate as positive if avg order value is above median
    global_median_spend = agg["AvgOrderValue"].median()
    agg["OrderAmountHikeFromlastYear"] = (agg["AvgOrderValue"] - global_median_spend).clip(lower=0)

    # Attach churn label
    agg = agg.merge(future, on="customer_unique_id", how="left")
    agg["Churn"] = (agg["_ordered_again"] != 1).astype(int)
    agg = agg.drop(columns=["_ordered_again", "first_order_date", "last_order_date"])

    logger.info(
        "Customers: %d | Churn rate: %.1f%%",
        len(agg),
        agg["Churn"].mean() * 100
    )

    # ── Feature mapping to pipeline schema ───────────────────────────────────
    # Fields available directly or via proxy
    agg["SatisfactionScore"] = agg["AvgReviewScore"].round(0).clip(1, 5).astype(int)
    agg["Complain"] = (agg["LowReviewCount"] > 0).astype(int)
    agg["CityTier"] = agg["CustomerState"].map(STATE_TIER).fillna(3).astype(int)
    agg["WarehouseToHome"] = agg["AvgFreightValue"].fillna(agg["AvgFreightValue"].median())
    agg["NumberOfAddress"] = 1  # Olist doesn't expose multiple delivery addresses
    agg["CashbackAmount"] = (agg["TotalSpend"] / agg["OrderCount"].clip(lower=1)).fillna(0)
    agg["CouponUsed"] = (agg["PreferredPaymentMode"] == "voucher").astype(int)

    # Proxies for missing fields (set to population medians to minimise noise)
    agg["HourSpendOnApp"]          = agg["TotalItemCount"].clip(upper=10)  # more items → more browsing
    agg["NumberOfDeviceRegistered"] = agg["DistinctSellers"].clip(1, 5)   # proxy: more sellers = broader usage
    agg["Gender"]                  = 0  # not available
    agg["MaritalStatus"]           = 0  # not available
    agg["PreferredLoginDevice"]    = 0  # not available

    # Encode categoricals
    le_payment = LabelEncoder()
    agg["PreferredPaymentMode"] = le_payment.fit_transform(agg["PreferredPaymentMode"].astype(str))
    le_cat = LabelEncoder()
    agg["PreferedOrderCat"] = le_cat.fit_transform(agg["PreferedOrderCat"].fillna("other").astype(str))

    agg = agg.rename(columns={"customer_unique_id": "CustomerID"})
    agg["CustomerID"] = agg["CustomerID"].astype(str)

    return agg


def engineer_behavioral_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build composite behavioral features using the same formulas as features.py.
    This keeps the segmentation and churn pipeline identical.
    """
    df = df.copy()

    df["EngagementScore"] = 0.5 * (
        df["HourSpendOnApp"] / df["HourSpendOnApp"].max()
    ) + 0.5 * (df["OrderCount"] / df["OrderCount"].max())

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


def get_olist_feature_sets() -> dict:
    """
    Feature sets for the Olist dataset.
    Clustering and churn feature lists mirror features.py where fields are available.
    """
    return {
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
            "PreferredPaymentMode",
            "PreferedOrderCat",
            "EngagementScore",
            "RecencySignal",
            "StickinessIndex",
            "SpendTrend",
            "SupportRiskScore",
            "DiscountSensitivity",
            "TenureStability",
            "WarehouseFriction",
        ],
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


def build_olist_pipeline(save: bool = True) -> pd.DataFrame:
    """
    Full Olist feature pipeline. Returns processed DataFrame ready for
    segmentation → churn → uplift pipeline stages.
    """
    logger.info("Loading Olist CSVs from %s", OLIST_PATH)
    dfs = load_olist_csvs()

    logger.info("Building order-level features...")
    orders = build_order_features(dfs)

    logger.info("Aggregating to customer level and engineering churn label...")
    df = engineer_customer_features(orders)

    logger.info("Engineering behavioral composite features...")
    df = engineer_behavioral_features(df)

    # Fill any remaining NaNs with column medians
    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    if save:
        os.makedirs(PROCESSED_PATH, exist_ok=True)
        out_path = os.path.join(PROCESSED_PATH, "features.parquet")
        df.to_parquet(out_path, index=False)
        logger.info("Saved Olist features to %s", out_path)

    logger.info("Olist pipeline complete. Shape: %s | Churn rate: %.1f%%",
                df.shape, df["Churn"].mean() * 100)
    return df


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    df = build_olist_pipeline(save=True)
    print("\nChurn distribution:")
    print(df["Churn"].value_counts())
    print("\nFeature summary:")
    print(df[[
        "OrderCount", "Tenure", "DaySinceLastOrder", "AvgOrderValue",
        "SatisfactionScore", "Complain", "EngagementScore", "RecencySignal",
    ]].describe().round(3))
