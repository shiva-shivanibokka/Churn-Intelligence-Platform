# ruff: noqa: I001  -- import order is load-bearing: common.py puts ../src on sys.path before project modules are imported
"""Recompute every section-4 figure in RESULTS.md from the committed artifacts.

Inputs (all committed):
  results/agent_eval_rows.csv             one row per (approach, seed, customer)
  results/agent_eval_raw_outputs.jsonl    the model replies / parsed outputs
  results/agent_eval_third_arm_rows.csv   arm B (full prompt, no tools), if present

This script exists so that no number in RESULTS.md section 4 is hand-computed.
It covers the subset-restricted and parseable-only figures that
``agent_eval.py`` does not write to ``agent_eval_summary_ci.csv``: the
Sleeping-Dog-only intervention rates, the never-intervene baseline, the
parseable-only agreement rates, and the by-customer-type agreement table.

ESTIMATORS. Two different estimators are reported side by side, and every
output row names the one it used, because they differ materially on the
parseable-only subset (0.530 vs 0.588):

  ratio          sum(numerator) / sum(denominator) over runs. Runs, not
                 customers, are the unit; customers contributing more
                 parseable replies get more weight.
  percustomer    unweighted mean over customers of each customer's own rate.
                 This is the estimator ``agent_eval.py`` uses, and the one the
                 bootstrap below is built for.

CIs. 95% percentile bootstrap, 2000 resamples, clustered on customer_id: a
resample draws customers with replacement and keeps all of a drawn customer's
runs. Seeds are not resampled -- the three seeds are reruns of the same 40
customers, so the customer is the independent unit, not the run. The CIs are
uncorrected for multiple comparisons.

Run:  python eval_sop/agent_section4_figures.py
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from common import OUT  # noqa: E402

N_BOOT = 2000
BOOT_SEED = 0
TYPES = ["Persuadable", "Lost Cause", "Sleeping Dog", "Sure Thing"]


def ratio(num: np.ndarray, den: np.ndarray) -> float:
    d = den.sum()
    return float(num.sum() / d) if d else float("nan")


def percustomer(cust: np.ndarray, num: np.ndarray, den: np.ndarray) -> float:
    """Unweighted mean over customers of num/den, skipping customers with den == 0."""
    vals = []
    for c in np.unique(cust):
        m = cust == c
        d = den[m].sum()
        if d:
            vals.append(num[m].sum() / d)
    return float(np.mean(vals)) if vals else float("nan")


def boot_ci(df: pd.DataFrame, num_col: str, den_col: str, estimator: str):
    """Clustered-on-customer percentile bootstrap for either estimator."""
    cust = df["customer_id"].values
    num = df[num_col].astype(float).values
    den = df[den_col].astype(float).values
    uniq = np.unique(cust)
    idx_of = {c: np.flatnonzero(cust == c) for c in uniq}
    point = ratio(num, den) if estimator == "ratio" else percustomer(cust, num, den)
    rng = np.random.default_rng(BOOT_SEED)
    vals = []
    for _ in range(N_BOOT):
        drawn = uniq[rng.integers(0, len(uniq), len(uniq))]
        rows = np.concatenate([idx_of[c] for c in drawn])
        # Re-label each drawn copy so percustomer weights copies, not originals.
        rep = np.concatenate([np.full(len(idx_of[c]), i) for i, c in enumerate(drawn)])
        v = ratio(num[rows], den[rows]) if estimator == "ratio" else percustomer(rep, num[rows], den[rows])
        if not np.isnan(v):
            vals.append(v)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return point, float(lo), float(hi)


def row(out, label, subset, df, num_col, den_col, estimator):
    point, lo, hi = boot_ci(df, num_col, den_col, estimator)
    out.append({"figure": label, "subset": subset, "estimator": estimator,
                "numerator": int(df[num_col].sum()), "denominator": int(df[den_col].sum()),
                "n_customers": int(df["customer_id"].nunique()),
                "estimate": point, "ci_lo": lo, "ci_hi": hi})


def main():
    rows_path = os.path.join(OUT, "agent_eval_rows.csv")
    d = pd.read_csv(rows_path)
    # Arm B (full prompt, tools removed) lives in its own file because it was run
    # separately -- adding it to agent_eval.py would have forced a re-run of the
    # expensive agent arm. Scored by the same agent_eval.score, so it concatenates.
    for extra in ("agent_eval_third_arm_rows.csv", "agent_eval_third_arm_armD_rows.csv"):
        p = os.path.join(OUT, extra)
        if os.path.exists(p):
            d = pd.concat([d, pd.read_csv(p)], ignore_index=True)
    d["one"] = 1.0
    d["valid_json"] = d["valid_json"].astype(bool)
    d["intervene"] = d["intervene"].astype(bool).astype(float)
    d["agrees_with_rule"] = d["agrees_with_rule"].astype(bool).astype(float)
    d["ref_intervene"] = d["ref_intervene"].astype(bool)

    # The never-intervene baseline is not an arm in agent_eval.py: it is the
    # degenerate policy "always output do not intervene". Its agreement with the
    # project rule is therefore 1 exactly on the customers the rule leaves alone.
    nv = d[d.approach == "rule_based"].copy()
    nv["approach"] = "never_intervene"
    nv["intervene"] = 0.0
    nv["agrees_with_rule"] = (~nv["ref_intervene"]).astype(float)
    nv["valid_json"] = True
    d = pd.concat([d, nv], ignore_index=True)

    out: list[dict] = []
    arms = ["agent", "tool_output_no_calls", "full_prompt_no_tools", "no_tools_llm",
            "rule_based", "never_intervene"]
    for ap in [a for a in arms if (d.approach == a).any()]:
        g = d[d.approach == ap]
        sd = g[g.customer_type == "Sleeping Dog"]
        row(out, "intervened_on_sleeping_dog", ap, sd, "intervene", "one", "ratio")
        row(out, "intervened_on_sleeping_dog", ap, sd, "intervene", "one", "percustomer")

        bad = g[g.customer_type.isin(["Lost Cause", "Sleeping Dog"])]
        row(out, "intervened_on_lost_cause_or_sleeping_dog", ap, bad, "intervene", "one", "ratio")
        row(out, "intervened_on_lost_cause_or_sleeping_dog", ap, bad, "intervene", "one", "percustomer")

        row(out, "valid_final_json", ap, g, "valid_json", "one", "ratio")
        row(out, "valid_final_json", ap, g, "valid_json", "one", "percustomer")

        # all runs: an unparseable reply counts as a disagreement
        row(out, "agrees_with_rule_all_runs", ap, g, "agrees_with_rule", "one", "ratio")
        row(out, "agrees_with_rule_all_runs", ap, g, "agrees_with_rule", "one", "percustomer")

        # parseable replies only
        p = g[g.valid_json]
        row(out, "agrees_with_rule_parseable_only", ap, p, "agrees_with_rule", "one", "ratio")
        row(out, "agrees_with_rule_parseable_only", ap, p, "agrees_with_rule", "one", "percustomer")

        row(out, "intervention_rate", ap, g, "intervene", "one", "ratio")

        for t in TYPES:
            pt = p[p.customer_type == t]
            if len(pt):
                row(out, f"agrees_with_rule_parseable_only::{t}", ap, pt, "agrees_with_rule", "one", "ratio")
                row(out, f"agrees_with_rule_parseable_only::{t}", ap, pt, "agrees_with_rule", "one", "percustomer")

    res = pd.DataFrame(out)
    res.to_csv(os.path.join(OUT, "agent_section4_figures.csv"), index=False)

    # ── Decomposition of the Sleeping-Dog contact rate ────────────────────────
    # Four arms, each pair holding all but one factor fixed. Paired on customer:
    # the same 10 Sleeping Dogs appear in every arm, so the difference is
    # bootstrapped over customers, not runs, using one shared set of draws.
    contrasts = []
    if (d.approach == "full_prompt_no_tools").any():
        pairs = [("agent", "full_prompt_no_tools", "the whole tool loop (prompt held fixed)"),
                 ("tool_output_no_calls", "full_prompt_no_tools", "having tool OUTPUT in context (no calling)"),
                 ("agent", "tool_output_no_calls", "function-CALLING itself (same information in both)"),
                 ("full_prompt_no_tools", "no_tools_llm", "the prompt (no tools in either)"),
                 ("agent", "no_tools_llm", "tools + prompt together (original confounded contrast)")]
        # Two outcome definitions, because they do not agree and reporting only one
        # would overstate the case: "Sleeping Dog" is the headline metric, and
        # "Lost Cause or Sleeping Dog" is the broader do-not-contact population.
        outcomes = {"sleeping_dog": ["Sleeping Dog"],
                    "lost_cause_or_sleeping_dog": ["Lost Cause", "Sleeping Dog"]}
        for oname, types in outcomes.items():
            sub = d[d.customer_type.isin(types)]
            per_cust = sub.groupby(["approach", "customer_id"])["intervene"].mean().unstack(0)
            rng = np.random.default_rng(BOOT_SEED)
            cust_ids = per_cust.index.values
            draws = [rng.integers(0, len(cust_ids), len(cust_ids)) for _ in range(N_BOOT)]
            for hi, lo, what in pairs:
                if hi not in per_cust or lo not in per_cust:
                    continue
                diff = (per_cust[hi] - per_cust[lo]).values
                b = [float(diff[idx].mean()) for idx in draws]
                ci_lo, ci_hi = np.percentile(b, [2.5, 97.5])
                contrasts.append({"outcome": oname, "isolates": what, "higher_arm": hi, "lower_arm": lo,
                                  "rate_higher": float(per_cust[hi].mean()), "rate_lower": float(per_cust[lo].mean()),
                                  "difference": float(diff.mean()), "ci_lo": float(ci_lo), "ci_hi": float(ci_hi),
                                  "n_customers": int(len(cust_ids)),
                                  "excludes_zero": bool(ci_lo > 0 or ci_hi < 0)})
        pd.DataFrame(contrasts).to_csv(os.path.join(OUT, "agent_sleeping_dog_decomposition.csv"), index=False)
        print("\n-- Do-not-contact decomposition (paired on customer, clustered bootstrap) --")
        print(pd.DataFrame(contrasts).round(4).to_string(index=False))

    # Where do the agent's unparseable replies sit?
    ag = d[(d.approach == "agent") & (~d.valid_json)]
    bytype = ag.groupby("customer_type").size().to_dict()
    # Across-seed std of the per-seed means (the +/- values quoted in RESULTS.md)
    per_seed = d[d.approach == "agent"].groupby("seed")[["valid_json", "agrees_with_rule"]].mean()
    info = {"n_boot": N_BOOT, "boot_seed": BOOT_SEED, "cluster": "customer_id",
            "n_customers": int(d["customer_id"].nunique()),
            "estimators": {"ratio": "sum(num)/sum(den) over runs",
                           "percustomer": "unweighted mean over customers of num/den"},
            "multiple_comparison_correction": "none; all intervals are uncorrected",
            "agent_unparseable_by_customer_type": bytype,
            "agent_across_seed_std": {"valid_json": float(per_seed["valid_json"].std()),
                                      "agrees_with_rule": float(per_seed["agrees_with_rule"].std())},
            "source": ["results/agent_eval_rows.csv"]}
    with open(os.path.join(OUT, "agent_section4_figures_info.json"), "w") as fh:
        json.dump(info, fh, indent=2)

    show = res[res.figure.isin(["intervened_on_sleeping_dog", "agrees_with_rule_parseable_only",
                                "agrees_with_rule_all_runs", "valid_final_json"])]
    print(show.to_string(index=False))
    print("\nagent unparseable by customer type:", bytype)
    print("across-seed std:", info["agent_across_seed_std"])


if __name__ == "__main__":
    main()
