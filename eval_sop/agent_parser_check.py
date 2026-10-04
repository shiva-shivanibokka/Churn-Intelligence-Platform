# ruff: noqa: I001  -- import order is load-bearing: common.py puts ../src on sys.path before project modules are imported
"""Parser-symmetry check for agent_eval.py.

agent_eval.py scores the agent with the project's own JSON parsing
(src/agent_loop.py: strip ``` fences, then json.loads of the whole reply),
but scored the no-tools LLM with a lenient regex parser (agent_eval.parse_json).

Two halves, because the two arms need different treatment:

1. ``--no-tools`` (needs the local model): re-runs the no-tools arm (same model,
   seeds, prompts) keeping the raw reply, and reports validity under BOTH
   parsers.
2. ``--agent`` (offline, no model): applies the LENIENT parser to the agent
   arm's parse failures recorded in ``results/agent_eval_raw_outputs.jsonl``,
   so the agent arm is also measured under both parsers.

   CAVEAT, and it is the whole story for this half: the project's parser stores
   only ``raw_response = raw[:500]`` (src/agent_loop.py), so every recorded
   agent failure is truncated at exactly 500 characters. A reply whose JSON
   object had not closed by character 500 cannot be re-parsed from the
   artifact at all -- not "fails the lenient parser", but *unknown*. This half
   therefore reports three counts: recovered, still-invalid, and
   INDETERMINATE-because-truncated. The full untruncated agent replies were
   not retained by the run, so closing that gap needs a re-run, not a re-parse.

Run:  python eval_sop/agent_parser_check.py --agent              # offline
      python eval_sop/agent_parser_check.py --no-tools --model qwen2.5:7b --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import json
import os

import pandas as pd

import agent_eval as ae
from common import OUT, ROOT


def project_parse(raw: str):
    """Exactly the parsing in src/agent_loop.generate_retention_action_agentic."""
    raw = raw or ""
    try:
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0].strip()
        return json.loads(raw)
    except (json.JSONDecodeError, IndexError):
        return None


def valid(o):
    return isinstance(o, dict) and ("do_not_intervene_reason" in o or "intervention_type" in o)


TRUNC_AT = 500  # src/agent_loop.py stores raw_response = raw[:500]


def check_agent_arm():
    """Apply the lenient parser to the agent arm's recorded parse failures.

    Offline: reads only committed artifacts. Returns the summary dict it writes.
    """
    path = os.path.join(OUT, "agent_eval_raw_outputs.jsonl")
    recs = [json.loads(line) for line in open(path, encoding="utf-8")]
    agent = [r for r in recs if r["approach"] == "agent"]
    rows = []
    for r in agent:
        o = r["output"]
        strict_ok = isinstance(o, dict) and "raw_response" not in o
        if strict_ok:
            rows.append({"seed": r["seed"], "customer_id": r["customer_id"], "strict_valid": True,
                         "lenient_valid": True, "truncated": False, "verdict": "strict_ok"})
            continue
        raw = (o or {}).get("raw_response", "") if isinstance(o, dict) else ""
        truncated = len(raw) >= TRUNC_AT
        lenient_ok = valid(ae.parse_json(raw))
        if lenient_ok:
            verdict = "recovered_by_lenient"          # a complete object closed before the cut
        elif truncated:
            verdict = "INDETERMINATE_truncated"       # cannot be decided from the artifact
        else:
            verdict = "invalid_under_both"
        rows.append({"seed": r["seed"], "customer_id": r["customer_id"], "strict_valid": False,
                     "lenient_valid": bool(lenient_ok), "truncated": truncated, "verdict": verdict,
                     "raw_len": len(raw)})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT, "agent_eval_agent_arm_parser_check.csv"), index=False)
    n = len(out)
    counts = out["verdict"].value_counts().to_dict()
    strict = int(out["strict_valid"].sum())
    recovered = int(counts.get("recovered_by_lenient", 0))
    indet = int(counts.get("INDETERMINATE_truncated", 0))
    invalid = int(counts.get("invalid_under_both", 0))
    summary = {
        "n_runs": n,
        "strict_parser_valid": strict,
        "lenient_parser_valid_lower_bound": strict + recovered,
        "lenient_parser_valid_upper_bound": strict + recovered + indet,
        "recovered_by_lenient": recovered,
        "invalid_under_both": invalid,
        "indeterminate_truncated_rows": indet,
        "truncation_limit_chars": TRUNC_AT,
        "note": ("src/agent_loop.py keeps only the first 500 characters of a reply it cannot parse, so "
                 f"{indet} of {n} agent runs cannot be re-parsed from the committed artifact. The lenient-parser "
                 f"validity for the agent arm is therefore bounded: {strict + recovered}/{n} to "
                 f"{strict + recovered + indet}/{n}, not a point estimate."),
        "source": ["results/agent_eval_raw_outputs.jsonl"],
    }
    with open(os.path.join(OUT, "agent_eval_agent_arm_parser_check.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5:7b")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--agent", action="store_true", help="offline: re-parse the agent arm's recorded failures")
    ap.add_argument("--no-tools", dest="no_tools", action="store_true", help="re-run the no-tools arm (needs the model)")
    args = ap.parse_args()
    if args.agent or not args.no_tools:
        check_agent_arm()
    if not args.no_tools:
        return
    df = pd.read_parquet(os.path.join(ROOT, "data", "processed", "uplift.parquet"))
    sample = pd.concat([df[df.CustomerType == t].sample(10, random_state=0) for t in ae.TYPES]).reset_index(drop=True)
    rows = []
    for seed in args.seeds:
        for _, r in sample.iterrows():
            raw = ae.no_tools(r.to_dict(), args.model, seed)
            rows.append({"seed": seed, "customer_id": r["CustomerID"], "customer_type": r["CustomerType"],
                         "valid_project_parser": valid(project_parse(raw)), "valid_lenient_parser": valid(ae.parse_json(raw)),
                         "raw": raw[:2000]})
    try:
        ae.unload(args.model)
    except Exception as e:
        print("unload failed:", e)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT, "agent_eval_no_tools_parser_check.csv"), index=False)
    print(out.groupby("seed")[["valid_project_parser", "valid_lenient_parser"]].mean())
    print(out[["valid_project_parser", "valid_lenient_parser"]].mean())


if __name__ == "__main__":
    main()
