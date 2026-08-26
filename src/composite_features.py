"""
The eight composite behavioural features, and the constants they normalise by.

Every dataset module (e-commerce, Olist, Cell2Cell) maps its raw columns onto
the same canonical names and then builds the same eight composites. That block
used to be copy-pasted into all three, and in all three it divided by
`df[col].max()` — a statistic of whatever rows happened to be in the frame.

That is fine exactly once, when the frame is the whole training set, and wrong
every other time. Scoring a single customer through the API made
`x / x.max()` collapse to `x / x`, so `EngagementScore`, `RecencySignal`,
`StickinessIndex`, `SpendTrend` and `WarehouseFriction` were all pinned to
**1.0 for every request**, whatever was in it. Five of the eight composites
carried no information at serving time, and nothing failed: the endpoint
returned a plausible probability, so a 60-month customer with five-star
satisfaction and 40 orders scored *lower* churn than a one-month complainer who
had not ordered in 45 days.

So the normalisers are now fitted once, on the training population, and saved
next to the models. Training numbers are unchanged — fitting on the full frame
and applying immediately is arithmetically what the old code did — but the
constants are now an artifact that inference can load, which is the only way
`/score` can mean anything.

`DiscountSensitivity`, `SupportRiskScore` and `TenureStability` are row-local
(a ratio, a weighted sum, a log) and need no fitted state. They are here anyway
so that all eight live in one place.
"""

import numpy as np
import pandas as pd

# Composite -> the columns whose maxima it divides by. Used to fit and to
# validate that a saved norms dict covers everything the transform needs.
NORM_COLUMNS = (
    "HourSpendOnApp",
    "OrderCount",
    "DaySinceLastOrder",
    "NumberOfDeviceRegistered",
    "NumberOfAddress",
    "OrderAmountHikeFromlastYear",
    "WarehouseToHome",
)

COMPOSITE_FEATURES = (
    "EngagementScore",
    "RecencySignal",
    "StickinessIndex",
    "SpendTrend",
    "SupportRiskScore",
    "DiscountSensitivity",
    "TenureStability",
    "WarehouseFriction",
)

# Denominators are floored rather than left raw. A column that is all zeros
# would otherwise divide by zero and produce NaN or inf, which CatBoost accepts
# and then quietly trains on.
_EPS = 1e-9


def fit_composite_norms(df: pd.DataFrame) -> dict:
    """
    Read the normalising constants off a training frame.

    Call this once, on the full training population, and persist the result
    alongside the models. Anything scoring new rows must load it rather than
    recomputing — recomputing on a single row is the bug this module exists for.
    """
    return {
        col: float(df[col].max()) if col in df.columns else 0.0
        for col in NORM_COLUMNS
    }


def add_composite_features(df: pd.DataFrame, norms: dict) -> pd.DataFrame:
    """
    Add the eight composites to `df` using pre-fitted `norms`.

    These mirror leading indicators documented in Uber/Spotify/Salesforce
    production systems: engagement depth (not just frequency, but how deeply
    customers use the product), recency decay, stickiness (cross-device usage
    signals commitment), support risk (complaint history is a leading churn
    indicator at Salesforce), spend trajectory, and discount sensitivity (heavy
    coupon use signals price-sensitivity, not loyalty).

    Mutates and returns `df`, matching how the dataset modules already call it.
    """
    missing = [c for c in NORM_COLUMNS if c not in norms]
    if missing:
        raise ValueError(
            f"Fitted norms are missing {missing}. They come from "
            "fit_composite_norms() on the training frame — see models/feature_norms.json."
        )

    def denom(col: str) -> float:
        return max(float(norms.get(col, 0.0)), _EPS)

    # Weighted composite: hours on app (depth) + order count (frequency).
    df["EngagementScore"] = 0.5 * (df["HourSpendOnApp"] / denom("HourSpendOnApp")) + 0.5 * (
        df["OrderCount"] / denom("OrderCount")
    )

    # Days since last order, normalised. Higher = more lapsed. Direct analog of
    # Spotify's "days since last stream" leading indicator.
    df["RecencySignal"] = df["DaySinceLastOrder"] / denom("DaySinceLastOrder")

    # Multiple devices = deep ecosystem integration (hard to leave). Multiple
    # addresses = loyalty across life events.
    df["StickinessIndex"] = (df["NumberOfDeviceRegistered"] + df["NumberOfAddress"]) / max(
        norms.get("NumberOfDeviceRegistered", 0.0) + norms.get("NumberOfAddress", 0.0), _EPS
    )

    # Year-over-year order amount change. Declining spend is a top-3 churn
    # predictor at Salesforce.
    df["SpendTrend"] = df["OrderAmountHikeFromlastYear"] / denom("OrderAmountHikeFromlastYear")

    # Complaint flag + inverted satisfaction (1 = best, 5 = worst), mirroring
    # Salesforce's "support sentiment" health score component. Row-local.
    df["SupportRiskScore"] = df["Complain"] * 0.6 + ((df["SatisfactionScore"] - 1) / 4) * 0.4

    # Coupon usage relative to order count: price-driven loyalty, not brand
    # loyalty — a leading churn indicator when discounts stop. Row-local.
    df["DiscountSensitivity"] = df["CouponUsed"] / (df["OrderCount"] + _EPS)

    # Log-transform dampens extreme tenure values. Row-local.
    df["TenureStability"] = np.log1p(df["Tenure"])

    # Delivery distance as friction.
    df["WarehouseFriction"] = df["WarehouseToHome"] / denom("WarehouseToHome")

    return df


def _self_check() -> None:
    """
    The property worth protecting: scoring one row must give the same composites
    as scoring it inside the training batch. That is exactly what the old
    recompute-per-call code got wrong, so it is what gets asserted.
    """
    train = pd.DataFrame(
        {
            "HourSpendOnApp": [1.0, 3.0, 5.0],
            "OrderCount": [2.0, 10.0, 40.0],
            "DaySinceLastOrder": [1.0, 15.0, 60.0],
            "NumberOfDeviceRegistered": [1, 3, 5],
            "NumberOfAddress": [1, 2, 4],
            "OrderAmountHikeFromlastYear": [5.0, 15.0, 30.0],
            "WarehouseToHome": [5.0, 15.0, 40.0],
            "Complain": [0, 1, 0],
            "SatisfactionScore": [5, 1, 3],
            "CouponUsed": [0.0, 4.0, 1.0],
            "Tenure": [2.0, 24.0, 60.0],
        }
    )
    norms = fit_composite_norms(train)
    batch = add_composite_features(train.copy(), norms)

    for i in range(len(train)):
        single = add_composite_features(train.iloc[[i]].copy(), norms)
        for feat in COMPOSITE_FEATURES:
            assert np.isclose(single[feat].iloc[0], batch[feat].iloc[i]), (
                f"row {i} feature {feat}: single-row {single[feat].iloc[0]} != "
                f"batch {batch[feat].iloc[i]}"
            )

    # And the failure mode it replaced: refitting on one row would have pinned
    # these to 1.0. Assert they are not all 1.0, or the check proves nothing.
    one_row = add_composite_features(train.iloc[[0]].copy(), norms)
    assert one_row["EngagementScore"].iloc[0] < 0.99, "single-row engagement collapsed to 1.0"
    assert one_row["RecencySignal"].iloc[0] < 0.99, "single-row recency collapsed to 1.0"

    # A degenerate all-zero column must not produce NaN or inf.
    zeros = train.copy()
    zeros["WarehouseToHome"] = 0.0
    out = add_composite_features(zeros, fit_composite_norms(zeros))
    assert np.isfinite(out["WarehouseFriction"]).all(), "zero column produced non-finite values"

    print("composite_features self-check passed")


if __name__ == "__main__":
    _self_check()
