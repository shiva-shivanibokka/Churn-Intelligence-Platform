# ruff: noqa: I001  -- import order is load-bearing: common.py puts ../src on sys.path before project modules are imported
"""Churn-model evaluation: reproduce the README holdout numbers, then add the
baselines the README's claims need (base rate, global model, logistic
regression) across several seeds, with bootstrap CIs.

Input: data/processed/segmented.parquet -- the committed artifact that
src/pipeline.py feeds into Stage 3 (per-segment churn models). Using it
means the segment labels are exactly the ones the committed models used.

Run:  python eval_sop/churn_eval.py
"""

from __future__ import annotations

import json
import logging
import os
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import OUT, ROOT, metrics_with_ci  # noqa: E402  (sets sys.path)

import churn_model  # noqa: E402  project code
from cell2cell_features import (  # noqa: E402
    clean_cell2cell,
    get_cell2cell_feature_sets,
    load_cell2cell,
)
from published_columns import CELL2CELL_SOURCE_COLUMNS  # noqa: E402

logging.basicConfig(level=logging.WARNING)
SEEDS = [42, 7, 13, 21, 99]
# CatBoost's results depend on thread_count (the project default is -1 = all cores), so the
# eval pins it. The seed-42 reproduction of the README still calls the project's own
# train_segment_model, which keeps thread_count=-1, exactly as the README numbers were made.
THREAD_COUNT = 2
FEATS = get_cell2cell_feature_sets()["churn_model"]

README_HOLDOUT_AUC = {  # copied from README.md "Per-segment churn models" table
    "At-Risk": 0.692, "Price Sensitive": 0.645, "Loyal Customers": 0.614,
    "Champions": 0.591, "Lapsed": 0.576,
}


def fit_catboost_calibrated(X, y, seed):
    """Exact recipe of churn_model.train_segment_model, with the seed exposed.

    80% train is split 75/25 into fit / calibration; early stopping and the
    isotonic map both use the calibration slice; class weight = neg/pos.
    """
    X_fit, X_cal, y_fit, y_cal = train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y)
    neg, pos = (y == 0).sum(), (y == 1).sum()
    params = churn_model.get_catboost_params(max(1.0, neg / pos))
    params["random_seed"] = seed
    params["thread_count"] = THREAD_COUNT
    clf = CatBoostClassifier(**params)
    clf.fit(X_fit, y_fit, eval_set=(X_cal, y_cal), early_stopping_rounds=50, use_best_model=True)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(clf.predict_proba(X_cal)[:, 1], y_cal)
    return churn_model.IsotonicCalibratedClassifier(clf, iso, len(y_cal))


# Minimum number of raw columns the reference arm needs before it is allowed to
# run. The real count is 54 (58 source columns, less CustomerID, less Churn, less
# the 3 leaky ones the builder drops, plus nothing added). A floor well below
# that still catches the failure that mattered -- the arm silently falling to a
# single column -- without breaking if the builder's drop list changes by one.
MIN_RAW_REFERENCE_COLUMNS = 40


def load_raw_reference(df):
    """Raw Cell2Cell columns for eval arm (5), aligned to `df`'s index.

    Returns ``(frame, column_names)``, or ``(None, [])`` when data/raw/ is not
    present -- in which case the arm is omitted from the results rather than
    reported as a degraded number.

    This replaces reading the raw columns out of segmented.parquet by position.
    That worked only for as long as the committed parquet happened to carry the
    source dataset, and when it stopped the arm did not fail: it scored AUC
    0.500 and wrote a one-element `raw_cols_reference` into the run info. A
    reference arm that can quietly become a coin flip is worse than one that is
    missing, so this raises on anything other than "the data is not here".
    """
    try:
        raw = clean_cell2cell(load_cell2cell())
    except FileNotFoundError as exc:
        print(
            "NOTE: raw Cell2Cell data not found, so eval arm (5) "
            "'reference_global_catboost_all_raw_cols' is OMITTED from this run. "
            "The committed parquets no longer publish the source columns, by "
            "design -- see src/published_columns.py. Download the dataset into "
            f"data/raw/cell2cell/ to reproduce that row. ({exc})"
        )
        return None, []

    raw_cols = sorted(
        c for c in raw.columns
        if c in CELL2CELL_SOURCE_COLUMNS and c not in ("CustomerID", "Churn")
    )
    if len(raw_cols) < MIN_RAW_REFERENCE_COLUMNS:
        raise RuntimeError(
            f"eval arm (5) needs the raw Cell2Cell columns but found only "
            f"{len(raw_cols)} ({raw_cols}). Refusing to report a reference "
            "number built on a fraction of the inputs it claims to use."
        )

    ids = df["CustomerID"].astype(str)
    raw = raw.copy()
    raw["CustomerID"] = raw["CustomerID"].astype(str)
    ref = (
        ids.to_frame()
        .merge(raw[["CustomerID", *raw_cols]], on="CustomerID", how="left")
        .set_index(df.index)
    )
    missing = int(ref[raw_cols[0]].isna().sum())
    if missing:
        raise RuntimeError(
            f"{missing} of {len(df)} rows in segmented.parquet have no matching "
            "CustomerID in the raw Cell2Cell data, so the reference arm would be "
            "trained on a different population than the other arms."
        )
    return ref, raw_cols

def run_seed(df, seed, raw_ref, raw_cols):
    # Per-segment 80/20 stratified split, exactly as the pipeline does it.
    test_idx, train_idx = [], []
    for seg in sorted(df["Segment"].unique()):
        sub = df.index[df["Segment"] == seg]
        tr, te = train_test_split(sub, test_size=0.20, random_state=seed, stratify=df.loc[sub, "Churn"])
        train_idx.extend(tr)
        test_idx.extend(te)
    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)
    tr, te = df.loc[train_idx], df.loc[test_idx]
    y_te = te["Churn"].values
    base_rate = float(tr["Churn"].mean())
    # Sensitivity: a per-segment base-rate forecast (each test row gets its segment's train churn rate)
    seg_rate = tr.groupby("Segment")["Churn"].mean()
    p_segrate = te["Segment"].map(seg_rate).values.astype(float)
    brier_segrate = float(np.mean((p_segrate - y_te) ** 2))
    preds = {}

    # (1) Project: per-segment CatBoost + isotonic
    p_seg = pd.Series(np.nan, index=te.index)
    p_seg_raw = pd.Series(np.nan, index=te.index)
    for seg in sorted(df["Segment"].unique()):
        trs, tes = tr[tr.Segment == seg], te[te.Segment == seg]
        m = fit_catboost_calibrated(trs[FEATS], trs["Churn"], seed)
        p_seg[tes.index] = m.predict_proba(tes[FEATS])[:, 1]
        p_seg_raw[tes.index] = m.base_clf.predict_proba(tes[FEATS])[:, 1]
    preds["per_segment_catboost_isotonic (project)"] = p_seg.values
    preds["per_segment_catboost_uncalibrated"] = p_seg_raw.values

    # (2) Global CatBoost + isotonic, same recipe, same 23 features, same train rows
    g = fit_catboost_calibrated(tr[FEATS], tr["Churn"], seed)
    preds["global_catboost_isotonic"] = g.predict_proba(te[FEATS])[:, 1]

    # (3) Global CatBoost + segment id as a feature
    feats_s = FEATS + ["RawSegment"]
    gs = fit_catboost_calibrated(tr[feats_s], tr["Churn"], seed)
    preds["global_catboost_isotonic+segment_id"] = gs.predict_proba(te[feats_s])[:, 1]

    # (4) Plain logistic regression, 23 features, no class weights, no calibration step
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    lr.fit(tr[FEATS], tr["Churn"])
    preds["logistic_regression"] = lr.predict_proba(te[FEATS])[:, 1]

    # (5) Reference: global CatBoost on all raw Cell2Cell columns (not what the project ships)
    #
    # The raw columns come from `raw_ref`, built from data/raw/, NOT from
    # segmented.parquet. They used to be read out of segmented.parquet by
    # POSITION -- `df.columns[:df.columns.index("Tenure")]` -- which was a silent
    # trap: once the committed parquets stopped publishing the source columns
    # that slice became ["MaritalStatus"], this arm scored AUC 0.500 instead of
    # 0.673, and nothing raised. If the raw data is absent the arm is omitted
    # from the output rather than reported as a number.
    if raw_ref is not None:
        gr = fit_catboost_calibrated(raw_ref.loc[train_idx, raw_cols], tr["Churn"], seed)
        preds["reference_global_catboost_all_raw_cols"] = gr.predict_proba(
            raw_ref.loc[test_idx, raw_cols]
        )[:, 1]

    # (6) Base rate
    preds["base_rate_constant"] = np.full(len(te), base_rate)

    rows = []
    for name, p in preds.items():
        if name == "base_rate_constant":
            from common import classification_metrics
            m = classification_metrics(y_te, p, base_rate)
            m = {k: float(v) for k, v in m.items()}
            for k in list(m):
                m[k + "_lo"] = m[k + "_hi"] = m[k]
            m["auc"] = m["auc_lo"] = m["auc_hi"] = 0.5
        else:
            m = metrics_with_ci(y_te, p, base_rate, seed=seed)
        m["brier_ref_segment_base_rate"] = brier_segrate
        m["bss_vs_segment_base_rate"] = 1.0 - m["brier"] / brier_segrate
        m.update(model=name, seed=seed, n_test=int(len(te)), n_train=int(len(tr)), thread_count=THREAD_COUNT,
                 test_churn_rate=float(y_te.mean()), train_base_rate=base_rate)
        # README-style statistic: unweighted mean of per-segment AUCs
        seg_aucs = {s: roc_auc_score(y_te[te.Segment.values == s], p[te.Segment.values == s])
                    for s in sorted(df.Segment.unique())} if name != "base_rate_constant" else {}
        m["mean_within_segment_auc"] = float(np.mean(list(seg_aucs.values()))) if seg_aucs else 0.5
        for s, a in seg_aucs.items():
            m[f"auc_within[{s}]"] = float(a)
        rows.append(m)

    # Paired bootstrap: per-segment (project) minus global, AUC and Brier, same test rows
    rng = np.random.default_rng(seed)
    a, b = preds["per_segment_catboost_isotonic (project)"], preds["global_catboost_isotonic"]
    d_auc, d_brier = [], []
    for _ in range(1000):
        idx = rng.integers(0, len(y_te), len(y_te))
        d_auc.append(roc_auc_score(y_te[idx], a[idx]) - roc_auc_score(y_te[idx], b[idx]))
        d_brier.append(np.mean((a[idx] - y_te[idx]) ** 2) - np.mean((b[idx] - y_te[idx]) ** 2))
    paired = {
        "seed": seed,
        "delta_auc_perseg_minus_global": float(roc_auc_score(y_te, a) - roc_auc_score(y_te, b)),
        "delta_auc_lo": float(np.percentile(d_auc, 2.5)), "delta_auc_hi": float(np.percentile(d_auc, 97.5)),
        "delta_brier_perseg_minus_global": float(np.mean((a - y_te) ** 2) - np.mean((b - y_te) ** 2)),
        "delta_brier_lo": float(np.percentile(d_brier, 2.5)), "delta_brier_hi": float(np.percentile(d_brier, 97.5)),
    }
    # Reliability table for the project model (10 equal-width bins)
    rel = []
    bins = np.clip((a * 10).astype(int), 0, 9)
    for bnum in range(10):
        msk = bins == bnum
        if msk.any():
            rel.append({"seed": seed, "bin": f"[{bnum/10:.1f},{(bnum+1)/10:.1f})", "n": int(msk.sum()),
                        "mean_pred": float(a[msk].mean()), "obs_rate": float(y_te[msk].mean())})
    return rows, paired, rel


def reproduce_committed():
    """Call the project's own train_segment_model (seed 42 hard-coded) and compare to README."""
    df = pd.read_parquet(os.path.join(ROOT, "data", "processed", "segmented.parquet"))
    out = []
    for seg in README_HOLDOUT_AUC:
        sub = df[df.Segment == seg]
        r = churn_model.train_segment_model(sub, sub["Churn"], seg, FEATS, mlflow_run=False)
        m = r["metrics"]
        out.append({"segment": seg, "readme_holdout_auc": README_HOLDOUT_AUC[seg],
                    "reproduced_holdout_auc": round(m["holdout_auc"], 4),
                    "holdout_brier_uncal": round(m["holdout_brier_uncalibrated"], 4),
                    "holdout_brier_cal": round(m["holdout_brier"], 4),
                    "n_test": m["n_test"], "test_churn_rate": float(r["y_test"].mean())})
    return out


def main():
    t0 = time.time()
    df = pd.read_parquet(os.path.join(ROOT, "data", "processed", "segmented.parquet")).reset_index(drop=True)
    raw_ref, raw_cols = load_raw_reference(df)

    repro = reproduce_committed()
    pd.DataFrame(repro).to_csv(os.path.join(OUT, "churn_reproduction_seed42.csv"), index=False)
    print(pd.DataFrame(repro).to_string())

    all_rows, paired, rel = [], [], []
    for s in SEEDS:
        r, p, rl = run_seed(df, s, raw_ref, raw_cols)
        all_rows += r
        paired.append(p)
        rel += rl
        print(f"seed {s} done ({time.time()-t0:.0f}s)")
    res = pd.DataFrame(all_rows)
    res.to_csv(os.path.join(OUT, "churn_metrics_by_seed.csv"), index=False)
    pd.DataFrame(paired).to_csv(os.path.join(OUT, "churn_perseg_vs_global_paired.csv"), index=False)
    pd.DataFrame(rel).to_csv(os.path.join(OUT, "churn_reliability_project_model.csv"), index=False)

    cols = ["auc", "pr_auc", "brier", "bss", "bss_vs_segment_base_rate", "ece", "mean_within_segment_auc"]
    summ = res.groupby("model")[cols].agg(["mean", "std"])
    summ.to_csv(os.path.join(OUT, "churn_summary_across_seeds.csv"))
    print(summ.round(4).to_string())
    print(pd.DataFrame(paired).round(4).to_string())
    with open(os.path.join(OUT, "churn_run_info.json"), "w") as fh:
        json.dump({"seeds": SEEDS, "catboost_thread_count": THREAD_COUNT, "n_rows": int(len(df)), "features": FEATS, "raw_cols_reference": raw_cols, "raw_reference_arm_ran": raw_ref is not None,
                   "runtime_s": round(time.time() - t0, 1)}, fh, indent=2)


if __name__ == "__main__":
    main()
