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

Almost nothing in this repository reads them. Of the 55 source columns in
``uplift.parquet``, two are needed: ``CustomerID`` as a join key, and ``Churn``
as the label — without the label the committed artifacts cannot be checked
against the reported metrics, which is the reason they are tracked at all.

So the rule is: **a column whose values come straight from the source dataset is
not published unless something actually needs it.** Derived columns —
composites, cluster assignments, model scores, uplift outputs — are this
repository's own work and are published as before.

**The rule is enforced against source column NAMES, which is weaker than it
sounds.** ``engineer_features`` in ``src/cell2cell_features.py`` is largely a
rename-and-clip table rather than a set of transformations, so several published
columns still carry a dropped source column's values under another name.
Measured against the raw CSV, the dropped source columns that stay essentially
recoverable are:

====================== ============================ =========================
dropped source column  recoverable from (published) rows exactly recoverable
====================== ============================ =========================
MonthsInService        Tenure, OrderCount           100.00%
RespondsToMailOffers   CouponUsed                   100.00%
Occupation             PreferedOrderCat             100.00% (bijective encode)
HandsetWebCapable      PreferredLoginDevice         100.00%
MonthlyRevenue         AvgOrderValue                 99.99%
CurrentEquipmentDays   DaySinceLastOrder             99.85%
RoamingCalls           WarehouseToHome               99.73%
UniqueSubs             NumberOfAddress               99.54%
Handsets               NumberOfDeviceRegistered      97.75%
CreditRating           SatisfactionScore             93.60%
MonthlyMinutes         HourSpendOnApp                85.40%
====================== ============================ =========================

So the honest figure is that the source information still present covers roughly
**13 of the 58 source columns** — the 2 allowlisted plus the 11 above, 8 of
them near-exactly — not 55, and not the "four signals" an earlier version of
this docstring claimed. The reduction is large; it is not total. Aliases that are
heavily lossy are not counted as recoverable: ``Complain`` keeps 54.89% of
``CustomerCareCalls`` (it is a ``> 3`` threshold), ``OrderAmountHikeFromlastYear``
43.95% of ``PercChangeRevenues`` (negatives are clipped away), and ``CityTier``
12.20% of ``IncomeGroup``.

Removing the aliases is not an option — they are the models' actual features.
The reduction available without breaking the models is the one taken here.

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

# data/raw/cell2cell/cell2celltrain.csv (58 columns). Both the train and the
# holdout CSV already use `CustomerID`; cell2cell_features.py also tolerates a
# `Customer_ID` spelling, but no file this dataset ships uses it.
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
})

# `MaritalStatus` was in the list above, on the stated grounds that app.py and
# the retention prompts read it. They do not — an adversarial review checked,
# and neither file mentions it. The only other references are api/serve.py, where
# it is a request-body field with a default and is never read from a parquet, and
# the e-commerce churn feature set, which this module does not enforce. So it was
# dropped. Recorded here rather than silently deleted, because the lesson is the
# general one: an allowlist is only as good as the claim attached to each entry,
# so each entry names its reader and the claim stays checkable.


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
