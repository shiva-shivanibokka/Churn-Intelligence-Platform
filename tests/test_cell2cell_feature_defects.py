"""Three dead or duplicated features in the Cell2Cell churn model.

All three were recorded in RESULTS.md before being fixed, and all three share a
shape: the pipeline ran, the figures looked plausible, and a feature that
contributed nothing sat inside the model. None of them raised anything.

  1. `clean_cell2cell` compared `Homeownership` against the string
     "known homeowner". The dataset holds "Known" and "Unknown", so every row
     became 0 and `PreferredPaymentMode` -- mapped from it -- was constant inside
     the churn feature list.
  2. `engineer_features` set `OrderCount = Tenure`, and both were in the
     `churn_model` feature list, so each per-segment model was fed the same
     56-valued column twice.
  3. `Gender` was hardcoded to 0 for all 51,047 rows and published in every
     tracked parquet while being named in no Cell2Cell feature set.

These tests work on a small synthetic frame shaped like the real schema, so they
run in a clone with no raw data.
"""
from __future__ import annotations

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cell2cell_features import (  # noqa: E402
    clean_cell2cell,
    engineer_features,
    get_cell2cell_feature_sets,
)


def make_raw(n: int = 60) -> pd.DataFrame:
    """A frame with the columns the three defects touch, and realistic values."""
    half = n // 2
    return pd.DataFrame({
        "CustomerID": [str(1000 + i) for i in range(n)],
        "Churn": [i % 2 for i in range(n)],
        # The real values. "known homeowner" appears nowhere in the dataset.
        "Homeownership": ["Known"] * half + ["Unknown"] * (n - half),
        "MonthsInService": [12 + (i % 24) for i in range(n)],
        "MonthlyRevenue": [50.0 + i for i in range(n)],
        "MonthlyMinutes": [300.0 + i for i in range(n)],
        "CustomerCareCalls": [float(i % 6) for i in range(n)],
        "CreditRating": ["1-Highest" if i % 2 else "4-Medium" for i in range(n)],
        "Handsets": [1.0 + (i % 3) for i in range(n)],
        "PercChangeRevenues": [float(i % 7) - 3 for i in range(n)],
        "CurrentEquipmentDays": [100.0 + i for i in range(n)],
        "RespondsToMailOffers": ["Yes" if i % 2 else "No" for i in range(n)],
        "RoamingCalls": [float(i % 5) for i in range(n)],
        # Deliberately a different cycle length from IncomeGroup below: at mod 3
        # both NumberOfAddress and CityTier came out identical, and the
        # duplicate-detection test then failed on the fixture rather than on
        # the code. The real data has no such coincidence.
        "UniqueSubs": [1 + (i % 5) for i in range(n)],
        "IncomeGroup": [str(1 + (i % 3)) for i in range(n)],
        "Occupation": ["Professional" if i % 2 else "Retired" for i in range(n)],
        "MaritalStatus": ["Yes" if i % 2 else "No" for i in range(n)],
        "HandsetWebCapable": ["Yes" if i % 2 else "No" for i in range(n)],
    })


@pytest.fixture(scope="module")
def engineered() -> pd.DataFrame:
    return engineer_features(clean_cell2cell(make_raw()))


# ---------------------------------------------------------------------------
# 1. Homeownership / PreferredPaymentMode
# ---------------------------------------------------------------------------


def test_homeownership_is_not_constant_after_cleaning():
    cleaned = clean_cell2cell(make_raw())
    assert cleaned["Homeownership"].nunique() == 2, (
        "Homeownership collapsed to one value — the string comparison in "
        "clean_cell2cell does not match the data's actual values again"
    )
    assert set(cleaned["Homeownership"].unique()) == {0, 1}


def test_known_homeowners_map_to_one_and_unknown_to_zero():
    """Pin the direction, not just that two values exist."""
    raw = make_raw(4)
    raw["Homeownership"] = ["Known", "Unknown", "known", " KNOWN "]
    cleaned = clean_cell2cell(raw)
    assert list(cleaned["Homeownership"]) == [1, 0, 1, 1]


def test_preferred_payment_mode_is_informative(engineered):
    assert engineered["PreferredPaymentMode"].nunique() == 2, (
        "PreferredPaymentMode is constant, so the churn model carries a feature "
        "that cannot affect any prediction"
    )


def test_cleaning_refuses_a_homeownership_column_it_cannot_decode():
    """The defect was silent. If the values ever change again it must not be."""
    raw = make_raw(10)
    raw["Homeownership"] = ["some-new-encoding"] * 10
    with pytest.raises(ValueError, match="collapsed to a single value"):
        clean_cell2cell(raw)


# ---------------------------------------------------------------------------
# 2. OrderCount duplicated Tenure inside the model
# ---------------------------------------------------------------------------


def test_churn_features_contain_no_duplicated_column(engineered):
    """The general form of the defect: no two features in the churn list may hold
    identical values. Catches a re-introduced alias under any name."""
    feats = get_cell2cell_feature_sets()["churn_model"]
    present = [f for f in feats if f in engineered.columns]
    dupes = []
    for i, a in enumerate(present):
        for b in present[i + 1:]:
            if engineered[a].equals(engineered[b]):
                dupes.append((a, b))
    assert not dupes, f"identical columns fed to the churn model twice: {dupes}"


def test_order_count_is_not_a_churn_model_feature():
    feats = get_cell2cell_feature_sets()["churn_model"]
    assert "OrderCount" not in feats
    assert "Tenure" in feats


def test_order_count_is_still_emitted_for_the_dashboard(engineered):
    """Dropping it from the model must not drop it from the data: Supabase's
    `customers` table and the dashboard both have an order-count field."""
    assert "OrderCount" in engineered.columns
    assert engineered["OrderCount"].equals(engineered["Tenure"]), (
        "the column's documented meaning on this dataset is tenure in months"
    )


def test_clustering_asks_for_tenure_not_order_count(engineered):
    """A rename, deliberately numerically inert: the clustering input matrix is
    unchanged, so the committed segments stay valid."""
    feats = get_cell2cell_feature_sets()["clustering"]
    assert "Tenure" in feats
    assert "OrderCount" not in feats
    old = [f if f != "Tenure" else "OrderCount" for f in feats]
    pd.testing.assert_frame_equal(
        engineered[feats].reset_index(drop=True),
        engineered[old].rename(columns={"OrderCount": "Tenure"}).reset_index(drop=True),
    )


# ---------------------------------------------------------------------------
# 3. Gender was a published constant
# ---------------------------------------------------------------------------


def test_gender_is_not_emitted_on_the_cell2cell_path(engineered):
    assert "Gender" not in engineered.columns, (
        "Gender is back. Cell2Cell has no gender field, so it can only be a "
        "constant, and a constant has no business in a published artifact"
    )


def test_no_churn_feature_is_constant(engineered):
    """The general form of defects 1 and 3 together."""
    feats = get_cell2cell_feature_sets()["churn_model"]
    constant = [
        f for f in feats
        if f in engineered.columns and engineered[f].nunique(dropna=False) <= 1
    ]
    assert not constant, f"features that cannot affect any prediction: {constant}"
