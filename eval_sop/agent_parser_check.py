# ruff: noqa: I001  -- import order is load-bearing: common.py puts ../src on sys.path before project modules are imported
"""Parser-symmetry check for agent_eval.py.

agent_eval.py scores the agent with the project's own JSON parsing
(src/agent_loop.py: strip ``` fences, then json.loads of the whole reply),
but scored the no-tools LLM with a lenient regex parser (agent_eval.parse_json).
This re-runs the no-tools arm (same model, seeds, prompts) keeping the raw
reply, and reports validity under BOTH parsers, so the comparison is symmetric.

Run:  python eval_sop/agent_parser_check.py --model qwen2.5:7b --seeds 0 1 2
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5:7b")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()
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
