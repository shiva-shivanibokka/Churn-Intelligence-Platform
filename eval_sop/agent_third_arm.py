# ruff: noqa: I001  -- import order is load-bearing: common.py puts ../src on sys.path before project modules are imported
"""Control arms B and D for the section-4 agent evaluation.

WHY THIS EXISTS. agent_eval.py compares two arms that differ in three ways at
once -- tool access, prompt content, and ~4.7 rounds of accumulated tool output
-- so its headline (Sleeping Dogs contacted in 20/30 runs with tools vs 1/30
without) cannot say which of the three caused the gap. These arms hold the
prompt fixed and strip the tool loop apart one layer at a time:

  A  agent                full SYSTEM_PROMPT_BATCH + 6 live tools + up to 5 rounds
  B  full_prompt_no_tools  full prompt verbatim, NO tools, NO tool output, one shot
  D  tool_output_no_calls  full prompt verbatim, NO tools, but the tool RESULTS
                           pasted into the user message, one shot
  C  no_tools_llm          TRUNCATED prompt (5-step sequence dropped), no tools

  A vs B  isolates the whole tool loop.      B vs C  isolates the prompt.
  D vs B  isolates having tool output in context (no function-calling).
  A vs D  isolates function-calling itself, given the same information.

Arms B and D are in this one script because they differ only in the user
message. Both use ``agent_loop.SYSTEM_PROMPT_BATCH`` **verbatim, with nothing
added or removed** -- it already carries the JSON contract and the
do-not-intervene instruction. It still tells the model to call tools that are
not supplied; that is the intended manipulation, not a bug.

HOW ARM D's TOOL OUTPUT IS BUILT, and the one caveat that matters. The run did
not store tool *results*, only the (tool, args) pairs, so they are recomputed.
Every tool in src/agent_tools.py is a deterministic pure function of
(df, playbook, args), so this reproduces them exactly -- it is a recomputation,
not an approximation. Arm D pastes the prompt's own prescribed 5-step sequence
(src/agent_loop.py:36-41) evaluated for that customer, through the same
leak-key stripping agent_eval.py applies, with the playbook queried for the
customer's TRUE top signed SHAP driver and ROI computed at the pipeline's own
flat $15 intervention cost.

  CAVEAT: that makes arm D's context *correct by construction*, whereas the live
  agent queried the playbook for the true top driver in only 71% of runs. Arm D
  is therefore a mildly FAVOURABLE reconstruction of what arm A had in context,
  not a replay of it. It bounds the effect of tool output in the direction that
  matters: if correct, clean tool output still drives the behaviour, noisier
  real output is unlikely to be the exculpating factor.

Scoring reuses agent_eval.score and agent_eval.parse_json unchanged, so B and D
are scored exactly as arm C was (lenient parser). Arm A remains on the project's
strict parser; see RESULTS.md section 4 "Parsing" for that asymmetry. Both arms
keep FULL untruncated replies, unlike arm A's 500-char cut.

Same 40 customers (10 per CustomerType, random_state=0) and same seeds {0,1,2}
as the original run. Sequential: one request at a time, one model job, resumable
checkpoint so a stalled local server does not lose completed work.

Run:  python eval_sop/agent_third_arm.py --arm B --model qwen2.5:7b --seeds 0 1 2
      python eval_sop/agent_third_arm.py --arm D --model qwen2.5:7b --seeds 0 1 2
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
import agent_tools  # noqa: E402  project code -- deterministic tool functions, used unmodified


ARMS = {"B": "full_prompt_no_tools", "D": "tool_output_no_calls"}
FLAT_COST = 15.0  # the pipeline's own flat intervention cost (NetROI assumption)


def base_user_message(row, clv=500.0):
    """The user message arm C gets, verbatim -- the shared starting point for B and D."""
    return (f"Generate a retention action plan for customer {row['CustomerID']}. "
            f"Their segment is '{row['Segment']}', churn probability is {row['ChurnProbability']:.1%}, "
            f"uplift score is {row['UpliftScore']:+.3f}, and customer type is '{row['CustomerType']}'. "
            f"Assume CLV = ${clv:.0f}.")


def _strip_leaks(d):
    """Exactly agent_eval.py's leak-key stripping, applied to a recomputed tool result."""
    if isinstance(d, dict):
        d = dict(d)
        for k in ae.LEAK_KEYS:
            d.pop(k, None)
    return d


def canonical_tool_results(row, df, playbook, clv=500.0):
    """Recompute the prompt's prescribed 5-step tool sequence for this customer.

    Order follows src/agent_loop.py:36-41. Deterministic: every tool is a pure
    function of (df, playbook, args). Leak keys stripped as the harness does.
    """
    cid = str(row["CustomerID"])
    feat, val = ae.top_driver(row)
    risk_factor = f"{'high' if val > 0 else 'low'} {feat}"
    steps = [
        ("get_top_churn_drivers", {"customer_id": cid},
         agent_tools.get_top_churn_drivers(df, cid)),
        ("search_retention_playbook", {"risk_factor": risk_factor},
         agent_tools.search_retention_playbook(playbook, risk_factor)),
        ("get_segment_benchmark", {"segment": str(row["Segment"])},
         agent_tools.get_segment_benchmark(df, str(row["Segment"]))),
        ("calculate_intervention_roi",
         {"uplift_score": float(row["UpliftScore"]), "clv": clv, "intervention_cost": FLAT_COST},
         agent_tools.calculate_intervention_roi(float(row["UpliftScore"]), clv, FLAT_COST)),
        ("lookup_customer_details", {"customer_id": cid},
         agent_tools.lookup_customer_details(df, cid)),
    ]
    return [(name, args, _strip_leaks(res)) for name, args, res in steps]


def run_one(arm, row, model, seed, df=None, playbook=None, clv=500.0):
    """Arm B or D: the project's complete batch prompt, verbatim, with no tools attached."""
    sys_prompt = agent_loop.SYSTEM_PROMPT_BATCH  # verbatim -- the whole point of these arms
    user = base_user_message(row, clv)
    tool_block = None
    if arm == "D":
        results = canonical_tool_results(row, df, playbook, clv)
        tool_block = "\n\n".join(
            f"{i}. {name}({json.dumps(args, default=str)}) returned:\n{json.dumps(res, indent=2, default=str)}"
            for i, (name, args, res) in enumerate(results, 1))
        user += ("\n\nThe tool calls in your sequence have already been run for you. "
                 "Their results are below; you have no tools to call, so use these and "
                 "output your final recommendation as JSON.\n\n" + tool_block)
    r = ae.ollama_chat(model, [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user}],
                       max_tokens=1500, temperature=ae.OllamaAsGroq.temperature, seed=seed)
    return (r.choices[0].message.content or ""), tool_block


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5:7b")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--limit", type=int, default=None, help="first N customers only (smoke test)")
    ap.add_argument("--arm", choices=["B", "D"], default="B",
                    help="B = no tools, no tool output; D = no tools, tool results pasted in")
    args = ap.parse_args()
    arm, approach = args.arm, ARMS[args.arm]
    # Arm B keeps the original unsuffixed filenames so its committed artifacts stay stable.
    tag = "" if arm == "B" else "_armD"

    df = pd.read_parquet(os.path.join(ROOT, "data", "processed", "uplift.parquet"))
    playbook = json.load(open(os.path.join(ROOT, "data", "playbook.json"), encoding="utf-8"))  # as agent_eval.py:264
    sample = pd.concat([df[df.CustomerType == t].sample(10, random_state=0) for t in ae.TYPES]).reset_index(drop=True)
    if args.limit:
        sample = sample.head(args.limit)

    ckpt = os.path.join(OUT, f"agent_eval_third_arm{tag}_checkpoint_{args.model.replace(':', '_')}.jsonl")
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
            tool_block = None
            try:
                raw, tool_block = run_one(arm, row, args.model, seed, df=df, playbook=playbook)
                infra = False
            except Exception as e:  # local server overloaded / timed out: infrastructure, not model, failure
                raw, infra = f"INFRA_ERROR: {e}", True
            lat = time.time() - ts
            out = ae.parse_json(raw) or {"raw": raw[:500]}
            s = ae.score(row, out, overflow=ae.BUDGET["overflow"])
            rec_row = {"approach": approach, "seed": seed, "customer_id": row["CustomerID"],
                       "customer_type": row["CustomerType"], "latency_s": lat, "infra_error": infra,
                       "llm_calls": ae.BUDGET["calls"], "max_est_prompt_tokens": ae.BUDGET["max_est_prompt_tokens"], **s}
            # Keep the FULL reply: the agent arm's 500-char truncation is what made its
            # parse failures un-adjudicable (RESULTS.md section 4). Do not repeat that here.
            rec_raw = {"approach": approach, "seed": seed, "customer_id": row["CustomerID"],
                       "output": out, "raw_response_full": raw}
            if tool_block is not None:
                rec_raw["pasted_tool_output"] = tool_block
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
    res.to_csv(os.path.join(OUT, f"agent_eval_third_arm{tag}_rows.csv"), index=False)
    with open(os.path.join(OUT, f"agent_eval_third_arm{tag}_raw_outputs.jsonl"), "w", encoding="utf-8") as fh:
        for r in raw_log:
            fh.write(json.dumps(r, default=str) + "\n")
    info = {"model": args.model, "arm_label": arm, "arm": approach,
            "system_prompt": "agent_loop.SYSTEM_PROMPT_BATCH verbatim, no tools attached",
            "user_message": "identical to agent_eval.no_tools",
            "parser": "agent_eval.parse_json (lenient) -- same as the no_tools_llm arm",
            "endpoint": "ollama local native /api/chat, num_ctx=8192, sequential",
            "seeds": args.seeds, "temperature": ae.OllamaAsGroq.temperature,
            "n_customers": int(sample["CustomerID"].nunique()),
            "sample": "10 per CustomerType from data/processed/uplift.parquet, pandas sample random_state=0",
            "runtime_s": round(time.time() - t0, 1), "n_infra_errors_excluded": n_infra,
            "max_est_prompt_tokens_over_runs": int(res["max_est_prompt_tokens"].max()),
            "raw_response_truncation": "none -- full replies retained"}
    if arm == "D":
        info["user_message"] = ("agent_eval.no_tools message PLUS the recomputed 5-step tool results "
                                "(src/agent_loop.py:36-41 order), leak keys stripped")
        info["tool_output_provenance"] = (
            "Recomputed, not replayed: the run stored (tool, args) but not results. Every tool in "
            "src/agent_tools.py is a deterministic pure function of (df, playbook, args), so this "
            "reproduces them exactly. Playbook queried for the customer's TRUE top signed SHAP driver; "
            f"ROI at the pipeline's flat ${FLAT_COST:.0f} intervention cost.")
        info["caveat"] = ("Arm D's context is correct by construction, whereas the live agent queried the "
                          "playbook for the true top driver in only 71% of runs. Arm D is therefore a mildly "
                          "FAVOURABLE reconstruction of arm A's context, not a replay of it.")
    with open(os.path.join(OUT, f"agent_eval_third_arm{tag}_run_info.json"), "w") as fh:
        json.dump(info, fh, indent=2)

    sd = res[res.customer_type == "Sleeping Dog"]
    print(f"\nSleeping Dog intervention: {sd.intervene.sum()}/{len(sd)} = {sd.intervene.mean():.3f}")
    print(f"valid_json: {res.valid_json.sum()}/{len(res)}")
    print(f"agrees_with_rule (all runs): {res.agrees_with_rule.sum()}/{len(res)}")
    print(res.groupby("customer_type")[["intervene", "valid_json", "agrees_with_rule"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
