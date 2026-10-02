"""Held-out evaluation of the project's uplift learners on a real randomized
experiment: the Hillstrom MineThatData e-mail challenge (64,000 customers,
randomized 1/3 Mens e-mail, 1/3 Womens e-mail, 1/3 no e-mail).

Treatment = received any e-mail (2/3 of rows). Primary outcome = visit within
two weeks; secondary = conversion.

The project's learners are called through its own functions in
src/uplift_model.py. Those functions assume the outcome is churn (bad) and
negate CausalML's mu1 - mu0. Feeding them "Churn" := 1 - visit therefore makes
their UpliftScore = increase in P(visit) from treatment -- the same convention,
with no edits to project code.

Run:  python eval_sop/uplift_hillstrom.py
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.request

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

from common import OUT, qini_coefficient, uplift_at_k, uplift_by_decile, uplift_curve_auuc  # noqa: E402

import uplift_model  # noqa: E402  project code

URL = "http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
SHA256 = "0e5893329d8b93cefecc571777672028290ab69865718020c78c7284f291aece"
DATA = os.path.join(os.path.dirname(__file__), "data", "hillstrom.csv")
SEEDS = [42, 7, 13, 21, 99]
N_BOOT = 500


def load():
    if not os.path.exists(DATA):
        os.makedirs(os.path.dirname(DATA), exist_ok=True)
        urllib.request.urlretrieve(URL, DATA)
    h = hashlib.sha256(open(DATA, "rb").read()).hexdigest()
    assert h == SHA256, f"hillstrom.csv sha256 mismatch: {h}"
    df = pd.read_csv(DATA)
    X = pd.get_dummies(df[["recency", "history", "mens", "womens", "newbie", "zip_code", "channel"]],
                       columns=["zip_code", "channel"], dtype=float)
    T = (df["segment"] != "No E-Mail").astype(int).values
    return df, X, T


def xgb_response(seed):
    return xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8,
                             eval_metric="auc", random_state=seed, n_jobs=-1)


def score_all(Xtr, Ttr, Ytr, Xte, seed):
    """Return dict name -> uplift score on test rows (higher = target first)."""
    feats = list(Xtr.columns)
    tr = Xtr.copy()
    tr["Treatment"] = Ttr
    tr["Churn"] = 1 - Ytr  # project convention: outcome to be reduced
    _, _, _, learners = uplift_model.compute_uplift_scores_causalml(tr, feats)
    up_t = -learners["t_learner"].predict(Xte.values).flatten()
    up_s = -learners["s_learner"].predict(Xte.values).flatten()
    ens = (up_t + up_s) / 2
    _, _, _, cl = uplift_model.compute_uplift_scores_custom(tr, feats)
    up_custom = cl["clf_t0"].predict_proba(Xte.values)[:, 1] - cl["clf_t1"].predict_proba(Xte.values)[:, 1]

    # Baselines
    resp = xgb_response(seed).fit(Xtr.values, Ytr)
    p_resp = resp.predict_proba(Xte.values)[:, 1]
    rng = np.random.default_rng(seed)
    return {
        "project_T+S_ensemble (as shipped)": ens,
        "project_T_learner (CausalML)": up_t,
        "project_S_learner (CausalML)": up_s,
        "project_T_learner (custom fallback)": up_custom,
        "project_ensemble_SIGN_FLIPPED (pre-fix bug)": -ens,
        "baseline_random": rng.random(len(Xte)),
        "baseline_churn_score (highest P(no visit))": 1 - p_resp,
        "baseline_response_model (highest P(visit))": p_resp,
    }


def eval_scores(y, t, scores, seed):
    rows = []
    rng = np.random.default_rng(1000 + seed)
    boots = [rng.integers(0, len(y), len(y)) for _ in range(N_BOOT)]
    for name, s in scores.items():
        point = {"qini": qini_coefficient(y, t, s), "auuc": uplift_curve_auuc(y, t, s),
                 "uplift@10%": uplift_at_k(y, t, s, 0.1), "uplift@30%": uplift_at_k(y, t, s, 0.3)}
        bvals = {k: [] for k in point}
        for idx in boots:
            bvals["qini"].append(qini_coefficient(y[idx], t[idx], s[idx]))
            bvals["auuc"].append(uplift_curve_auuc(y[idx], t[idx], s[idx]))
            bvals["uplift@10%"].append(uplift_at_k(y[idx], t[idx], s[idx], 0.1))
            bvals["uplift@30%"].append(uplift_at_k(y[idx], t[idx], s[idx], 0.3))
        r = {"model": name, "seed": seed, "n_test": int(len(y))}
        for k, v in point.items():
            r[k] = v
            r[k + "_lo"], r[k + "_hi"] = np.percentile(bvals[k], [2.5, 97.5])
        rows.append(r)
    return rows


def main():
    t0 = time.time()
    df, X, T = load()
    all_rows, deciles = [], []
    for outcome in ["visit", "conversion"]:
        Y = df[outcome].values
        for seed in SEEDS:
            strat = T * 2 + Y
            idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=0.3, random_state=seed, stratify=strat)
            scores = score_all(X.iloc[idx_tr], T[idx_tr], Y[idx_tr], X.iloc[idx_te], seed)
            yte, tte = Y[idx_te], T[idx_te]
            rows = eval_scores(yte, tte, scores, seed)
            ate = yte[tte == 1].mean() - yte[tte == 0].mean()
            for r in rows:
                r["outcome"] = outcome
                r["test_ate"] = float(ate)
                r["pred_mean_uplift"] = float(np.mean(scores[r["model"]])) if r["model"].startswith("project") else np.nan
            all_rows += rows
            for name in ["project_T+S_ensemble (as shipped)", "baseline_churn_score (highest P(no visit))",
                         "project_ensemble_SIGN_FLIPPED (pre-fix bug)"]:
                for d in uplift_by_decile(yte, tte, scores[name]):
                    d.update(model=name, seed=seed, outcome=outcome)
                    deciles.append(d)
            print(f"{outcome} seed {seed} done ({time.time()-t0:.0f}s)")
    res = pd.DataFrame(all_rows)
    res.to_csv(os.path.join(OUT, "uplift_hillstrom_by_seed.csv"), index=False)
    pd.DataFrame(deciles).to_csv(os.path.join(OUT, "uplift_hillstrom_deciles.csv"), index=False)
    summ = res.groupby(["outcome", "model"])[["qini", "auuc", "uplift@10%", "uplift@30%", "test_ate"]].agg(["mean", "std"])
    summ.to_csv(os.path.join(OUT, "uplift_hillstrom_summary.csv"))
    print(summ.round(5).to_string())
    with open(os.path.join(OUT, "uplift_hillstrom_run_info.json"), "w") as fh:
        json.dump({"url": URL, "sha256": SHA256, "n_rows": int(len(df)), "treated_share": float(T.mean()),
                   "features": list(X.columns), "seeds": SEEDS, "test_size": 0.3, "n_boot": N_BOOT,
                   "note": "project learners use xgboost random_state=42 hard-coded in src/uplift_model.py; "
                           "seed varies the train/test split and the baselines",
                   "runtime_s": round(time.time() - t0, 1)}, fh, indent=2)


if __name__ == "__main__":
    main()
