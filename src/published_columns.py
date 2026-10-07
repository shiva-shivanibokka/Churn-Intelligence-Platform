"""What may be committed to git, and what may not.

The processed parquets under ``data/processed/`` are tracked on purpose — the
deployed dashboard and ``migrate_to_supabase.py`` load them so the app runs from
a clone without re-running the pipeline (see ``.gitignore``). That is a
deliberate trade and it stays.

What was *not* deliberate is which columns went along for the ride. Every tracked
parquet carried the source dataset's own feature values — for Cell2Cell, 55 of
``uplift.parquet``'s 103 columns were verbatim source columns — and the
Cell2Cell data has no licence granting redistribution. It was compiled by the
Teradata Center for CRM at Duke University (Neslin et al., 2006); the copy in
circulation is a Kaggle mirror whose licence field reads "Unknown". The raw CSV
is correctly gitignored, so publishing the same values inside a parquet was an
accident of carrying the whole frame to disk, not a decision.

Nothing in this repository reads most of them. Of the 55 source columns in
``uplift.parquet``, exactly three are named by any code that reads the file:
``CustomerID``, ``Churn`` and ``MaritalStatus``. The rest are inert.

So the rule here is: **a column whose values come straight from the source
dataset is not published unless something actually needs it.** Derived columns —
composites, cluster assignments, model scores, uplift outputs — are this
repository's own work and are published as before.

Two source signals do survive, and saying so is the point of this module rather
than something to leave implicit:

* ``MaritalStatus`` and ``Churn`` are source values, kept because readers name
  them. ``Churn`` is the label; without it the committed artifacts cannot be
  checked against the reported metrics at all.
* ``Tenure`` and ``OrderCount`` are *derived* names and so survive the rule
  automatically, but on the Cell2Cell path both are exact copies of the source
  column ``MonthsInService`` (verified by comparison against the raw CSV). The
  reduction is therefore large, not total: four source signals instead of 55.

Olist is not listed below. ``src/olist_features.py`` builds its frame by
aggregating nine normalised tables into per-customer summaries, so its output
holds no verbatim row-level copy of a source table.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

# ─── Source schemas, as the raw files declare them ───────────────────────────
#
# Hardcoded because a clone has no raw data — the whole point is that it does not
# need any. `tests/test_published_columns.py` re-reads the real headers when the
# raw files happen to be present, so these cannot drift silently.

# data/raw/cell2cell/cell2celltrain.csv (58 columns; Customer_ID is renamed to
# CustomerID by src/cell2cell_features.py).
CELL2CELL_SOURCE_COLUMNS: frozenset[str] = frozenset({
    "CustomerID",
    "Churn", "MonthlyRevenue", "MonthlyMinutes", "TotalRecurringCharge",
    "DirectorAssistedCalls", "OverageMinutes", "RoamingCalls", "PercChangeMinutes",
    "PercChangeRevenues", "DroppedCalls", "BlockedCalls", "UnansweredCalls",
    "CustomerCareCalls", "ThreewayCalls", "ReceivedCalls", "OutboundCalls",
    "InboundCalls", "PeakCallsInOut", "OffPeakCallsInOut", "DroppedBlockedCalls",
    "CallForwardingCalls", "CallWaitingCalls", "MonthsInService", "UniqueSubs",
    "ActiveSubs", "ServiceArea", "Handsets", "HandsetModels", "CurrentEquipmentDays",
    "AgeHH1", "AgeHH2", "ChildrenInHH", "HandsetRefurbished", "HandsetWebCapable",
    "TruckOwner", "RVOwner", "Homeownership", "BuysViaMailOrder",
    "RespondsToMailOffers", "OptOutMailings", "NonUSTravel", "OwnsComputer",
    "HasCreditCard", "RetentionCalls", "RetentionOffersAccepted",
    "NewCellphoneUser", "NotNewCellphoneUser", "ReferralsMadeBySubscriber",
    "IncomeGroup", "OwnsMotorcycle", "AdjustmentsToCreditRating",
    "HandsetPrice", "MadeCallToRetentionTeam", "CreditRating", "PrizmCode",
    "Occupation", "MaritalStatus",
})

# data/raw/E Commerce Dataset.xlsx, sheet "E Comm" (20 columns).
ECOMMERCE_SOURCE_COLUMNS: frozenset[str] = frozenset({
    "CustomerID", "Churn", "Tenure", "PreferredLoginDevice", "CityTier",
    "WarehouseToHome", "PreferredPaymentMode", "Gender", "HourSpendOnApp",
    "NumberOfDeviceRegistered", "PreferedOrderCat", "SatisfactionScore",
    "MaritalStatus", "NumberOfAddress", "Complain",
    "OrderAmountHikeFromlastYear", "CouponUsed", "OrderCount",
    "DaySinceLastOrder", "CashbackAmount",
})

# Only cell2cell is enforced, and the reason is specific rather than an oversight.
#
# `src/cell2cell_features.py` maps the raw telecom columns onto the e-commerce
# schema's names (`MonthsInService` becomes `Tenure`, and so on) before anything
# is modelled, so on that path the models train entirely on renamed and derived
# columns and the raw names are inert passengers. All 23 churn features, all 13
# clustering features and all 9 uplift features survive the trim — checked, not
# assumed.
#
# On the e-commerce path the opposite is true: the source column names ARE the
# feature names. Enforcing the same rule there would drop 17 of the 26 churn
# features, 5 of 13 clustering features and 3 of 9 uplift features, which breaks
# the pipeline rather than protecting anything. `ECOMMERCE_SOURCE_COLUMNS` is
# kept above because the drift test uses it and because the next reader deserves
# to see that the question was asked, but it is deliberately not enforced. The
# exposure of that dataset is a real and separate problem; stripping columns the
# models need is not its solution.
SOURCE_COLUMNS: dict[str, frozenset[str]] = {
    "cell2cell": CELL2CELL_SOURCE_COLUMNS,
}

# Source columns that ARE published, because code that reads the parquets names
# them. Keep this set as small as the code allows; adding to it means publishing
# more of someone else's dataset.
PUBLISHABLE_SOURCE_COLUMNS: frozenset[str] = frozenset({
    # Join key for Supabase and the dashboard.
    "CustomerID",
    # The label. Without it the committed artifacts cannot be checked against
    # the reported AUC/Brier/churn-rate figures, which defeats the reason the
    # parquets are tracked at all.
    "Churn",
    # Read by app.py and the retention prompts.
    "MaritalStatus",
})


def forbidden_columns(dataset: str) -> frozenset[str]:
    """Source columns that must never reach a tracked parquet for *dataset*."""
    return SOURCE_COLUMNS.get(dataset, frozenset()) - PUBLISHABLE_SOURCE_COLUMNS


def restrict_for_publication(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Drop source-dataset columns nothing reads, preserving column order.

    Returns the frame unchanged for a dataset with no declared source schema, so
    adding a dataset cannot silently start dropping its columns — it just does
    not gain the protection until its schema is declared.
    """
    drop = forbidden_columns(dataset)
    if not drop:
        return df
    keep = [c for c in df.columns if c not in drop]
    return df[keep]


def offending_columns(columns: Iterable[str], dataset: str) -> list[str]:
    """Which of *columns* this dataset is not allowed to publish."""
    drop = forbidden_columns(dataset)
    return sorted(c for c in columns if c in drop)
