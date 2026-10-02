"""(a) Overlap / positivity diagnostics for the observational 'treatment' used
on Cell2Cell, and (b) the sign-flip counterfactual: what the targeted set and
the pipeline's own ROI estimate would be if the documented sign bug were still
present.

Input: data/processed/uplift.parquet (committed pipeline output, Cell2Cell).

The sign bug: CausalML returns mu1 - mu0; the shipped code negates it
(src/uplift_model.py:187,192). Before the fix the raw value was used, so the
buggy UpliftScore is exactly -UpliftScore for both learners and their average.
This is a recomputation of the counterfactual on today's fitted learners, not
a replay of a historical run.

Run:  python eval_sop/positivity_signflip.py
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import OUT, ROOT  # noqa: E402

import uplift_model  # noqa: E402
from cell2cell_features import get_cell2cell_feature_sets  # noqa: E402

UFEATS = get_cell2cell_feature_sets()["uplift_model"]


def positivity(df):
    out = {}
    ct = pd.crosstab(df["Complain"], df["Treatment"])
    out["crosstab_complain_x_treatment"] = {f"Complain={i}": {f"T={j}": int(ct.loc[i, j]) for j in ct.columns} for i in ct.index}
    out["P(T=1|Complain=1)"] = float(df.loc[df.Complain == 1, "Treatment"].mean())
    out["P(T=1|Complain=0)"] = float(df.loc[df.Complain == 0, "Treatment"].mean())
    out["share_rows_with_Complain=1"] = float(df.Complain.mean())
    X = df[UFEATS].values
    T = df["Treatment"].values
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    props = {}
    for name, model in {
        "logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "hist_gbm": HistGradientBoostingClassifier(random_state=0),
    }.items():
        e = cross_val_predict(model, X, T, cv=cv, method="predict_proba")[:, 1]
        props[name] = e
        out[f"propensity[{name}]"] = {
            "cv_auc_T_from_X": float(roc_auc_score(T, e)),
            "share_e<0.05": float((e < 0.05).mean()),
            "share_e>0.95": float((e > 0.95).mean()),
            "share_e>0.99": float((e > 0.99).mean()),
            "share_outside_[0.1,0.9]": float(((e < 0.1) | (e > 0.9)).mean()),
            "mean_e_given_Complain=1": float(e[df.Complain.values == 1].mean()),
            "min_e_given_Complain=1": float(e[df.Complain.values == 1].min()),
            "mean_e_given_Complain=0": float(e[df.Complain.values == 0].mean()),
        }
    smd = {}
    for f in UFEATS + ["ChurnProbability"]:
        a, b = df.loc[df.Treatment == 1, f], df.loc[df.Treatment == 0, f]
        smd[f] = float((a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2))
    out["standardized_mean_difference_T1_vs_T0"] = smd
    hist = []
    e = props["hist_gbm"]
    for lo in np.arange(0, 1, 0.1):
        m = (e >= lo) & (e < lo + 0.1 if lo < 0.9 else e <= 1)
        hist.append({"bin": f"[{lo:.1f},{lo+0.1:.1f})", "n_treated": int((m & (T == 1)).sum()), "n_control": int((m & (T == 0)).sum())})
    pd.DataFrame(hist).to_csv(os.path.join(OUT, "cell2cell_propensity_histogram.csv"), index=False)
    return out


def direction_check_is_not_causal(df):
    """The pipeline's validate_uplift_direction() 'passes' for scores that carry no uplift information."""
    d = df.copy()
    res = {}
    for name, s in {
        "shipped UpliftScore": d["UpliftScore"],
        "sign-flipped UpliftScore (bug)": -d["UpliftScore"],
        "minus ChurnProbability (pure risk score, no causal content)": -d["ChurnProbability"],
        "random noise": pd.Series(np.random.default_rng(0).random(len(d)), index=d.index),
    }.items():
        d["_s"] = s.values
        res[name] = uplift_model.validate_uplift_direction(d, uplift_col="_s")
    return res


def sign_flip(df, clv=500.0, cost=15.0):
    out = {}
    versions = {"fixed (shipped)": df["UpliftScore"].values, "buggy (pre-fix sign)": -df["UpliftScore"].values}
    sets, tops = {}, {}
    for name, u in versions.items():
        d = df[["CustomerID", "ChurnProbability", "Churn", "Treatment"]].copy()
        d["UpliftScore"] = u
        d["CustomerType"] = [uplift_model.classify_customer_type(a, b) for a, b in zip(d.UpliftScore, d.ChurnProbability)]
        d = uplift_model.estimate_intervention_roi(d, avg_clv=clv, intervention_cost=cost)
        p = d[d.CustomerType == "Persuadable"]
        sets[name] = set(p.CustomerID)
        tops[name] = list(p.sort_values("InterventionPriority").CustomerID.head(100))
        fixed_u = pd.Series(df["UpliftScore"].values, index=df.CustomerID)
        out[name] = {
            "type_counts": d.CustomerType.value_counts().to_dict(),
            "n_persuadable": int(len(p)),
            "n_persuadable_with_positive_reported_roi": int((p.NetROI > 0).sum()),
            "pipeline_reported_net_roi_sum_over_persuadables_usd": float(p.NetROI.sum()),
            "pipeline_reported_net_roi_sum_over_positive_roi_persuadables_usd": float(p.loc[p.NetROI > 0, "NetROI"].sum()),
            "fixed_model_net_roi_sum_for_this_set_usd": float((fixed_u.loc[p.CustomerID].values * clv - cost).sum()),
            "observed_churn_rate_in_set": float(p.Churn.mean()),
            "mean_ChurnProbability_in_set": float(p.ChurnProbability.mean()),
        }
    a, b = sets["fixed (shipped)"], sets["buggy (pre-fix sign)"]
    out["persuadable_overlap"] = {"intersection": len(a & b), "jaccard": len(a & b) / max(1, len(a | b))}
    out["top100_priority_overlap"] = len(set(tops["fixed (shipped)"]) & set(tops["buggy (pre-fix sign)"]))
    out["spearman_fixed_vs_buggy_score"] = -1.0
    out["assumptions"] = {"avg_clv": clv, "intervention_cost": cost,
                          "thresholds": "uplift>=0.05 and churn_prob>=0.30 (classify_customer_type defaults)"}
    return out


def main():
    df = pd.read_parquet(os.path.join(ROOT, "data", "processed", "uplift.parquet"))
    res = {"n_rows": int(len(df)), "uplift_features": UFEATS,
           "treatment_definition": "Treatment = (Complain==1) | (CouponUsed>0); on Cell2Cell Complain = CustomerCareCalls>3, CouponUsed = RespondsToMailOffers",
           "positivity": positivity(df),
           "direction_check_on_various_scores": direction_check_is_not_causal(df),
           "sign_flip": sign_flip(df),
           "mean_uplift_score": float(df.UpliftScore.mean()),
           "share_uplift_ge_0.05": float((df.UpliftScore >= 0.05).mean()),
           "share_uplift_ge_0.03_breakeven_at_clv500_cost15": float((df.UpliftScore >= 0.03).mean())}
    with open(os.path.join(OUT, "cell2cell_positivity_signflip.json"), "w") as fh:
        json.dump(res, fh, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    main()
