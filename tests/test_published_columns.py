"""The tracked parquets must not redistribute the source dataset.

`data/processed/*.parquet` is committed on purpose so the deployed dashboard runs
from a clone. The source dataset's own feature values went along with it by
accident: the pipeline wrote whatever frame it happened to be holding. Cell2Cell
has no licence permitting redistribution, and almost none of those columns is
read by anything.

These tests fail against the parquets as they were committed before
`src/published_columns.py` existed, which is the point — they are the check that
the trim happened and that a later pipeline run cannot quietly undo it.
"""
from __future__ import annotations

import json
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from published_columns import (  # noqa: E402
    CELL2CELL_SOURCE_COLUMNS,
    ECOMMERCE_SOURCE_COLUMNS,
    PUBLISHABLE_SOURCE_COLUMNS,
    SOURCE_COLUMNS,
    forbidden_columns,
    offending_columns,
    restrict_for_publication,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED = os.path.join(ROOT, "data", "processed")
META = os.path.join(ROOT, "models", "pipeline_meta.json")

TRACKED_PARQUETS = [
    "features.parquet",
    "scored.parquet",
    "segmented.parquet",
    "uplift.parquet",
]


def _committed_dataset() -> str:
    with open(META, encoding="utf-8") as fh:
        return json.load(fh)["dataset"]


# ---------------------------------------------------------------------------
# The defect: tracked artifacts carrying the source dataset
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", TRACKED_PARQUETS)
def test_tracked_parquet_publishes_no_source_feature_columns(name):
    path = os.path.join(PROCESSED, name)
    if not os.path.exists(path):
        pytest.skip(f"{name} not present")
    dataset = _committed_dataset()
    cols = pd.read_parquet(path).columns
    offending = offending_columns(cols, dataset)
    assert not offending, (
        f"{name} publishes {len(offending)} column(s) taken straight from the "
        f"{dataset} source dataset, which this repository has no licence to "
        f"redistribute: {offending[:8]}{'...' if len(offending) > 8 else ''}"
    )


def test_the_label_and_join_key_survive_the_trim():
    """Trimming too much is also a failure: the point of tracking these files is
    that the reported metrics can be re-derived from them."""
    path = os.path.join(PROCESSED, "uplift.parquet")
    if not os.path.exists(path):
        pytest.skip("uplift.parquet not present")
    cols = set(pd.read_parquet(path).columns)
    for needed in ("CustomerID", "Churn", "Segment", "ChurnProbability", "RiskTier"):
        assert needed in cols, f"{needed} was dropped; metrics can no longer be checked"


def test_readme_metrics_columns_all_survive():
    """scripts/readme_metrics.py runs in CI with --check against uplift.parquet."""
    path = os.path.join(PROCESSED, "uplift.parquet")
    if not os.path.exists(path):
        pytest.skip("uplift.parquet not present")
    cols = set(pd.read_parquet(path).columns)
    for needed in ("Churn", "ChurnProbability", "CustomerType", "RiskTier", "Segment"):
        assert needed in cols


def test_supabase_migration_columns_all_survive():
    """migrate_to_supabase.py intersects its column map with what is present, so a
    dropped column degrades silently into a missing dashboard field rather than
    an error. Pin the map instead."""
    path = os.path.join(PROCESSED, "uplift.parquet")
    if not os.path.exists(path):
        pytest.skip("uplift.parquet not present")
    cols = set(pd.read_parquet(path).columns)
    expected = {
        "CustomerID", "Segment", "ChurnProbability", "RiskTier", "UpliftScore",
        "CustomerType", "NetROI", "ROIPositive", "InterventionPriority",
        "UMAP_1", "UMAP_2", "Tenure", "SatisfactionScore", "DaySinceLastOrder",
        "HourSpendOnApp", "Complain", "OrderCount", "CashbackAmount", "Churn",
        "TopSHAPFeatures", "GMM_Prob_Seg0", "GMM_Prob_Seg1", "GMM_Prob_Seg2",
        "GMM_Prob_Seg3", "GMM_Prob_Seg4",
    }
    missing = sorted(expected - cols)
    assert not missing, f"the Supabase dashboard would lose these fields: {missing}"


# ---------------------------------------------------------------------------
# The declared schemas must match reality where reality is available
# ---------------------------------------------------------------------------


def test_declared_cell2cell_schema_matches_the_raw_csv_when_present():
    """The hardcoded schema exists because a clone has no raw data. When the raw
    file IS present, it must agree — otherwise the protection silently narrows."""
    raw_dir = os.path.join(ROOT, "data", "raw", "cell2cell")
    candidates = [
        os.path.join(raw_dir, n)
        for n in ("cell2celltrain.csv", "Cell2CellTrain.csv", "cell2cell.csv")
    ]
    path = next((p for p in candidates if os.path.exists(p)), None)
    if path is None:
        pytest.skip("raw Cell2Cell CSV not present (expected in a clone)")
    header = set(pd.read_csv(path, nrows=0).columns)
    header = {("CustomerID" if c == "Customer_ID" else c) for c in header}
    assert header == set(CELL2CELL_SOURCE_COLUMNS), (
        "declared Cell2Cell schema has drifted from the CSV; "
        f"only in CSV: {sorted(header - set(CELL2CELL_SOURCE_COLUMNS))}, "
        f"only declared: {sorted(set(CELL2CELL_SOURCE_COLUMNS) - header)}"
    )


def test_declared_ecommerce_schema_matches_the_workbook_when_present():
    path = os.path.join(ROOT, "data", "raw", "E Commerce Dataset.xlsx")
    if not os.path.exists(path):
        pytest.skip("raw e-commerce workbook not present (expected in a clone)")
    header = set(pd.read_excel(path, sheet_name="E Comm", nrows=0).columns)
    assert header == set(ECOMMERCE_SOURCE_COLUMNS), (
        "declared e-commerce schema has drifted from the workbook; "
        f"only in workbook: {sorted(header - set(ECOMMERCE_SOURCE_COLUMNS))}, "
        f"only declared: {sorted(set(ECOMMERCE_SOURCE_COLUMNS) - header)}"
    )


# ---------------------------------------------------------------------------
# The restriction helper itself
# ---------------------------------------------------------------------------


def test_restrict_drops_source_columns_and_keeps_derived_ones():
    df = pd.DataFrame({
        "CustomerID": [1, 2],
        "Churn": [0, 1],
        "MonthlyRevenue": [10.0, 20.0],     # source, nothing reads it
        "DroppedCalls": [1, 2],             # source, nothing reads it
        "EngagementScore": [0.5, 0.6],      # derived here
        "ChurnProbability": [0.1, 0.9],     # derived here
    })
    out = restrict_for_publication(df, "cell2cell")
    assert list(out.columns) == [
        "CustomerID", "Churn", "EngagementScore", "ChurnProbability"
    ]
    # Values of the kept columns are untouched.
    pd.testing.assert_frame_equal(out, df[list(out.columns)])


def test_restrict_is_a_no_op_for_an_undeclared_dataset():
    """Adding a dataset must not silently start dropping its columns."""
    df = pd.DataFrame({"a": [1], "MonthlyRevenue": [2.0]})
    out = restrict_for_publication(df, "olist")
    assert list(out.columns) == ["a", "MonthlyRevenue"]


def test_publishable_source_columns_are_a_subset_of_every_enforced_schema():
    for dataset, schema in SOURCE_COLUMNS.items():
        unknown = PUBLISHABLE_SOURCE_COLUMNS - set(schema)
        assert not unknown, (
            f"{sorted(unknown)} is allowed through but is not a {dataset} source "
            "column, so the allowance does nothing and misleads the next reader"
        )


def test_the_ecommerce_path_is_deliberately_not_enforced():
    """Enforcing it would drop 17 of the 26 e-commerce churn features. The schema
    constant exists for the drift test and for the record; it must stay out of
    SOURCE_COLUMNS until something maps those columns the way Cell2Cell does."""
    assert "ecommerce" not in SOURCE_COLUMNS
    assert forbidden_columns("ecommerce") == frozenset()
    assert len(ECOMMERCE_SOURCE_COLUMNS) == 20


def test_the_cell2cell_model_features_all_survive_the_trim():
    """The whole trim rests on this: cell2cell_features.py renames the raw telecom
    columns onto the e-commerce schema before modelling, so no feature the models
    use is a source column name."""
    sys.path.insert(0, os.path.join(ROOT, "src"))
    from cell2cell_features import get_cell2cell_feature_sets  # noqa: PLC0415

    forbidden = forbidden_columns("cell2cell")
    for name, feats in get_cell2cell_feature_sets().items():
        dropped = [f for f in feats if f in forbidden]
        assert not dropped, f"feature set {name!r} would lose {dropped}"


def test_leaky_columns_are_forbidden_from_publication():
    """The builder drops RetentionCalls & co as leakage. They must also never be
    published, which is a different guarantee from not being trained on."""
    forbidden = forbidden_columns("cell2cell")
    for leaky in ("RetentionCalls", "RetentionOffersAccepted", "MadeCallToRetentionTeam"):
        assert leaky in forbidden
