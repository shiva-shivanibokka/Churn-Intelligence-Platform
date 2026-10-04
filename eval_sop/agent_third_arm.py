# ruff: noqa: I001  -- import order is load-bearing: common.py puts ../src on sys.path before project modules are imported
"""Third agent arm: the FULL project prompt with the tools removed.

WHY THIS EXISTS. agent_eval.py compares two arms that differ in three ways at
once -- tool access, prompt content, and ~4.7 rounds of accumulated tool output
-- so its headline (Sleeping Dogs contacted in 20/30 runs with tools vs 1/30
without) cannot say which of the three caused the gap. This arm holds the prompt
fixed and removes only the tools:

  A  agent          full SYSTEM_PROMPT_BATCH + 6 tools + up to 5 rounds
  B  full_prompt_no_tools   full SYSTEM_PROMPT_BATCH verbatim, NO tools, one shot   <-- this script
  C  no_tools_llm    TRUNCATED prompt (5-step sequence dropped), no tools, one shot

  A vs B isolates the tools.   B vs C isolates the prompt.

The system prompt here is ``agent_loop.SYSTEM_PROMPT_BATCH`` used **verbatim,
with nothing added or removed** -- it already carries the JSON contract and the
do-not-intervene instruction, so no patching is needed. It still tells the model
to call tools that are not supplied; that is the intended manipulation, not a
bug. The user message is identical to the one arm C gets (agent_eval.no_tools),
so B differs from C only in the system prompt and from A only in tool access.

Scoring reuses agent_eval.score and agent_eval.parse_json unchanged, so arm B is
scored exactly as arm C was (lenient parser). Arm A remains on the project's
strict parser; see RESULTS.md section 4 "Parsing" for that asymmetry.

Same 40 customers (10 per CustomerType, random_state=0) and same seeds {0,1,2}
as the original run. Sequential: one request at a time, one model job, resumable
checkpoint so a stalled local server does not lose completed work.

Run:  python eval_sop/agent_third_arm.py --model qwen2.5:7b --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import json
import os
import time

import pandas as pd

import agent_eval as ae
from common import OUT, ROOT  # noqa: E402

import agent_loop  # noqa: E402  project code


def full_prompt_no_tools(row, model, seed, clv=500.0):
    """Arm B: the project's complete batch prompt, verbatim, with no tools attached."""
    sys_prompt = agent_loop.SYSTEM_PROMPT_BATCH  # verbatim -- the whole point of this arm
    user = (f"Generate a retention action plan for customer {row['CustomerID']}. "
            f"Their segment is '{row['Segment']}', churn probability is {row['ChurnProbability']:.1%}, "
            f"uplift score is {row['UpliftScore']:+.3f}, and customer type is '{row['CustomerType']}'. "
            f"Assume CLV = ${clv:.0f}.")
    r = ae.ollama_chat(model, [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user}],
                       max_tokens=1500, temperature=ae.OllamaAsGroq.temperature, seed=seed)
    return r.choices[0].message.content or ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5:7b")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--limit", type=int, default=None, help="first N customers only (smoke test)")
    args = ap.parse_args()

    df = pd.read_parquet(os.path.join(ROOT, "data", "processed", "uplift.parquet"))
    sample = pd.concat([df[df.CustomerType == t].sample(10, random_state=0) for t in ae.TYPES]).reset_index(drop=True)
    if args.limit:
        sample = sample.head(args.limit)

    ckpt = os.path.join(OUT, f"agent_eval_third_arm_checkpoint_{args.model.replace(':', '_')}.jsonl")
    done = {}
    if os.path.exists(ckpt):
        for line in open(ckpt, encoding="utf-8"):
            rec = json.loads(line)
            done[(rec["seed"], rec["customer_id"])] = rec

    rows, raw_log = [], []
    t0 = time.time()
    for seed in args.seeds:
        for i, r in sample.iterrows():
            row = r.to_dict()
            key = (seed, row["CustomerID"])
            if key in done:
                rows.append(done[key]["row"])
                raw_log.append(done[key]["raw"])
                continue
            ae.reset_budget()
            ts = time.time()
            try:
                raw = full_prompt_no_tools(row, args.model, seed)
                infra = False
            except Exception as e:  # local server overloaded / timed out: infrastructure, not model, failure
                raw, infra = f"INFRA_ERROR: {e}", True
            lat = time.time() - ts
            out = ae.parse_json(raw) or {"raw": raw[:500]}
            s = ae.score(row, out, overflow=ae.BUDGET["overflow"])
            rec_row = {"approach": "full_prompt_no_tools", "seed": seed, "customer_id": row["CustomerID"],
                       "customer_type": row["CustomerType"], "latency_s": lat, "infra_error": infra,
                       "llm_calls": ae.BUDGET["calls"], "max_est_prompt_tokens": ae.BUDGET["max_est_prompt_tokens"], **s}
            # Keep the FULL reply: the agent arm's 500-char truncation is what made its
            # parse failures un-adjudicable (RESULTS.md section 4). Do not repeat that here.
            rec_raw = {"approach": "full_prompt_no_tools", "seed": seed, "customer_id": row["CustomerID"],
                       "output": out, "raw_response_full": raw}
            rows.append(rec_row)
            raw_log.append(rec_raw)
            with open(ckpt, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"seed": seed, "customer_id": row["CustomerID"], "row": rec_row, "raw": rec_raw}, default=str) + "\n")
            print(f"seed {seed} customer {i+1}/{len(sample)} ({time.time()-t0:.0f}s)", flush=True)

    try:
        ae.unload(args.model)  # keep_alive: 0
        print("model unloaded (keep_alive=0)")
    except Exception as e:
        print("unload failed:", e)

    res = pd.DataFrame(rows)
    n_infra = int(res["infra_error"].sum())
    print(f"infra errors excluded from scoring: {n_infra}")
    res = res[~res["infra_error"]]
    res.to_csv(os.path.join(OUT, "agent_eval_third_arm_rows.csv"), index=False)
    with open(os.path.join(OUT, "agent_eval_third_arm_raw_outputs.jsonl"), "w", encoding="utf-8") as fh:
        for r in raw_log:
            fh.write(json.dumps(r, default=str) + "\n")
    with open(os.path.join(OUT, "agent_eval_third_arm_run_info.json"), "w") as fh:
        json.dump({"model": args.model, "arm": "full_prompt_no_tools",
                   "system_prompt": "agent_loop.SYSTEM_PROMPT_BATCH verbatim, no tools attached",
                   "user_message": "identical to agent_eval.no_tools",
                   "parser": "agent_eval.parse_json (lenient) -- same as the no_tools_llm arm",
                   "endpoint": "ollama local native /api/chat, num_ctx=8192, sequential",
                   "seeds": args.seeds, "temperature": ae.OllamaAsGroq.temperature,
                   "n_customers": int(sample["CustomerID"].nunique()),
                   "sample": "10 per CustomerType from data/processed/uplift.parquet, pandas sample random_state=0",
                   "runtime_s": round(time.time() - t0, 1), "n_infra_errors_excluded": n_infra,
                   "raw_response_truncation": "none -- full replies retained"}, fh, indent=2)

    sd = res[res.customer_type == "Sleeping Dog"]
    print(f"\nSleeping Dog intervention: {sd.intervene.sum()}/{len(sd)} = {sd.intervene.mean():.3f}")
    print(f"valid_json: {res.valid_json.sum()}/{len(res)}")
    print(f"agrees_with_rule (all runs): {res.agrees_with_rule.sum()}/{len(res)}")
    print(res.groupby("customer_type")[["intervene", "valid_json", "agrees_with_rule"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
