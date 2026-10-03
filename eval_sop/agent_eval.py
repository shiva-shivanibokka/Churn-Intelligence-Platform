"""Small, programmatic evaluation of the retention agent against two baselines.

Approaches (same 40 customers, stratified 10 per CustomerType, Cell2Cell):
  agent     -- the project's Python ReAct loop, src/agent_loop.py
               generate_retention_action_agentic(), unmodified, with its Groq
               client swapped for a local Ollama model through the
               OpenAI-compatible endpoint (free, local). Tools = src/agent_tools.py.
  no_tools  -- same model, same system prompt's JSON contract, same user
               message, but no tools.
  rule      -- deterministic: intervene iff CustomerType == 'Persuadable' and
               NetROI > 0 (i.e. ranks by uplift x CLV - cost); intervention from
               the playbook entry for the customer's top signed SHAP driver.

Scoring is programmatic (no LLM judge). There is NO ground-truth retention
outcome for these customers -- Cell2Cell has no randomized intervention -- so
'decision agreement' is agreement with the project's own decision rule, a
consistency check, not a measure of business value.

Eval-harness change (does not modify src/): lookup_customer_details returns
`actually_churned` -- the realized label -- to the LLM. It is stripped here so
the agent cannot read the answer. See RESULTS.md.

Run:  python eval_sop/agent_eval.py [--model qwen2.5:7b] [--seeds 0 1 2] [--n 40]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time

import numpy as np
import pandas as pd

from common import OUT, ROOT  # noqa: E402

import agent_loop  # noqa: E402
import agent_tools  # noqa: E402

OLLAMA_NATIVE = "http://localhost:11434/api/chat"
NUM_CTX = 8192  # hard context budget for this eval (coordinator constraint)
TYPES = ["Persuadable", "Sure Thing", "Lost Cause", "Sleeping Dog"]
COST_MID = {"low": 3.0, "medium": 17.5, "high": 75.0}
CHARS_PER_TOKEN_EST = 3.0  # conservative (over-counts tokens); checked against Ollama prompt_eval_count on cold calls

# Per-run budget bookkeeping, reset by the caller before each agent / no-tools run.
BUDGET = {"max_est_prompt_tokens": 0, "overflow": False, "calls": 0, "cold_calls": []}


class _Obj:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _to_native(messages):
    """OpenAI-style message list -> Ollama native: tool-call arguments as dicts."""
    out = []
    for m in messages:
        m = dict(m)
        if m.get("tool_calls"):
            tcs = []
            for tc in m["tool_calls"]:
                f = tc["function"] if isinstance(tc, dict) else tc.function.__dict__
                args = f["arguments"]
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                tcs.append({"function": {"name": f["name"], "arguments": args}})
            m["tool_calls"] = tcs
        m.pop("tool_call_id", None)
        out.append(m)
    return out


def ollama_chat(model, messages, tools=None, max_tokens=1500, temperature=0.7, seed=0):
    """One request to the native Ollama API with num_ctx fixed at NUM_CTX.

    Returns an object shaped like an OpenAI ChatCompletion so src/agent_loop.py runs unchanged.
    finish_reason is 'tool_calls' when the model called tools (this mirrors Ollama's own
    OpenAI-compatible endpoint; the native API always says 'stop').
    """
    import urllib.request
    body = {"model": model, "messages": _to_native(messages), "stream": False,
            "options": {"num_ctx": NUM_CTX, "num_predict": max_tokens, "temperature": temperature, "seed": seed}}
    if tools:
        body["tools"] = tools
    est = (len(json.dumps(body["messages"])) + len(json.dumps(tools or []))) / CHARS_PER_TOKEN_EST
    BUDGET["calls"] += 1
    BUDGET["max_est_prompt_tokens"] = max(BUDGET["max_est_prompt_tokens"], int(est))
    if est + max_tokens > NUM_CTX:
        BUDGET["overflow"] = True  # counted as a failure by score(); the call still runs (Ollama would truncate)
    req = urllib.request.Request(OLLAMA_NATIVE, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=900))
    if BUDGET["calls"] == 1:  # first call of a run has a cold prefix -> prompt_eval_count is the real prompt size
        BUDGET["cold_calls"].append({"est_tokens": int(est), "prompt_eval_count": r.get("prompt_eval_count")})
    msg = r.get("message", {})
    tcs = [
        _Obj(id=f"call_{i}", type="function",
             function=_Obj(name=tc["function"]["name"], arguments=json.dumps(tc["function"].get("arguments", {}))))
        for i, tc in enumerate(msg.get("tool_calls") or [])
    ]
    finish = "tool_calls" if tcs else ("length" if r.get("done_reason") == "length" else "stop")
    return _Obj(choices=[_Obj(message=_Obj(content=msg.get("content", ""), tool_calls=tcs or None), finish_reason=finish)])


class _Completions:
    def __init__(self, seed, temperature):
        self.seed, self.temperature = seed, temperature

    def create(self, model, messages, tools=None, tool_choice=None, max_tokens=1500, **kw):
        return ollama_chat(model, messages, tools=tools, max_tokens=max_tokens,
                           temperature=kw.get("temperature", self.temperature), seed=self.seed)


class _Chat:
    def __init__(self, comp):
        self.completions = comp


class OllamaAsGroq:
    """Drop-in for groq.Groq(api_key=...) used inside agent_loop.run_agentic_loop."""

    seed = 0
    temperature = 0.7

    def __init__(self, api_key=None):
        self.chat = _Chat(_Completions(self.seed, self.temperature))


def unload(model):
    import urllib.request
    req = urllib.request.Request("http://localhost:11434/api/generate", data=json.dumps({"model": model, "keep_alive": 0}).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=120))


LEAK_KEYS = ("actually_churned", "churn", "Churn")


def leak_free_execute_tool(orig):
    def run(name, args, df, playbook):
        r = orig(name, args, df, playbook)
        if isinstance(r, dict):
            for k in LEAK_KEYS:
                r.pop(k, None)
        return r
    return run


def parse_json(raw: str):
    raw = raw or ""
    raw = re.sub(r"<think>[\s\S]*?</think>", "", raw)
    if "```json" in raw:
        raw = raw.split("```json")[1].split("```")[0]
    elif "```" in raw:
        raw = raw.split("```")[1].split("```")[0]
    m = re.search(r"\{[\s\S]*\}", raw)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def top_driver(row):
    d = json.loads(row["TopSHAPFeatures"]) if isinstance(row["TopSHAPFeatures"], str) else {}
    if not d:
        return None, 0.0
    k = max(d, key=lambda f: abs(d[f]))
    return k, d[k]


def rule_policy(row, playbook):
    if row["CustomerType"] != "Persuadable" or row["NetROI"] <= 0:
        return {"do_not_intervene_reason": f"CustomerType={row['CustomerType']}, NetROI={row['NetROI']:.2f}"}
    feat, val = top_driver(row)
    pb = agent_tools.search_retention_playbook(playbook, f"{'high' if val > 0 else 'low'} {feat}")
    return {"primary_risk_reason": f"Top SHAP driver {feat} ({val:+.3f})",
            "intervention_type": pb.get("intervention_type"), "channel": pb.get("channel"),
            "timing": pb.get("timing"), "intervention_cost_estimate": pb.get("cost_tier")}


def no_tools(row, model, seed, clv=500.0):
    sys_prompt = agent_loop.SYSTEM_PROMPT_BATCH.split("After calling all relevant tools,")[0].split("Use the available tools")[0]
    sys_prompt += ("You have NO tools. Decide from the information given.\n\nOutput your recommendation as valid JSON ONLY (no other text):\n"
                   + agent_loop.SYSTEM_PROMPT_BATCH.split("output your final recommendation as valid JSON ONLY (no other text):")[1])
    user = (f"Generate a retention action plan for customer {row['CustomerID']}. "
            f"Their segment is '{row['Segment']}', churn probability is {row['ChurnProbability']:.1%}, "
            f"uplift score is {row['UpliftScore']:+.3f}, and customer type is '{row['CustomerType']}'. "
            f"Assume CLV = ${clv:.0f}.")
    r = ollama_chat(model, [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user}],
                    max_tokens=1500, temperature=OllamaAsGroq.temperature, seed=seed)
    return r.choices[0].message.content or ""


def cost_of(tier):
    t = str(tier or "").lower()
    for k, v in COST_MID.items():
        if k in t:
            return v
    return np.nan


def _close(x, y, tol=0.01):
    try:
        return abs(float(x) - float(y)) < tol
    except (TypeError, ValueError):
        return False


def score(row, out, trace=None, overflow=False):
    feat, _ = top_driver(row)
    # A run whose prompt would not fit the 8k context is a failure regardless of what came back.
    valid = (not overflow) and isinstance(out, dict) and ("do_not_intervene_reason" in out or "intervention_type" in out)
    intervene = bool(valid and "intervention_type" in out and "do_not_intervene_reason" not in out)
    ref = bool(row["CustomerType"] == "Persuadable" and row["NetROI"] > 0)
    reason = str(out.get("primary_risk_reason", "")) if isinstance(out, dict) else ""
    cost = cost_of(out.get("intervention_cost_estimate")) if intervene else 0.0
    ev = float(row["UpliftScore"]) * 500.0
    tools = [t["tool"] for t in (trace or [])]
    return {
        "valid_json": valid, "intervene": intervene, "ref_intervene": ref,
        "agrees_with_rule": bool(valid and intervene == ref),
        "intervened_on_lost_cause_or_sleeping_dog": bool(intervene and row["CustomerType"] in ("Lost Cause", "Sleeping Dog")),
        "cites_true_top_shap_driver": bool(intervene and feat and feat.lower() in reason.lower()),
        "cost_exceeds_model_expected_value": bool(intervene and cost > ev),
        "model_estimated_net_value_usd": float(ev - cost) if intervene else 0.0,
        "n_tool_calls": len(tools), "called_drivers_tool": "get_top_churn_drivers" in tools,
        "called_roi_tool": "calculate_intervention_roi" in tools,
        "budget_overflow": bool(overflow),
        # Did the agent's playbook query target the customer's actual top driver, and did it pass its true uplift to the ROI tool?
        "playbook_query_matches_top_driver": bool(feat and any(t["tool"] == "search_retention_playbook" and feat.lower() in str(t["args"]).lower() for t in (trace or []))),
        "roi_tool_used_true_uplift": bool(any(t["tool"] == "calculate_intervention_roi" and _close(t["args"].get("uplift_score"), row["UpliftScore"])
                                             for t in (trace or []) if isinstance(t.get("args"), dict))),
    }


def reset_budget():
    BUDGET.update(max_est_prompt_tokens=0, overflow=False, calls=0)


def is_infra(err: str) -> bool:
    e = str(err or "").lower()
    return any(k in e for k in ("timed out", "connection", "urlopen", "http error 5", "remote end closed"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5:7b")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--limit", type=int, default=None, help="only the first K scenarios (for timing)")
    ap.add_argument("--scenarios-only", action="store_true", help="write the scenario set + rule baseline, call no LLM")
    args = ap.parse_args()

    df = pd.read_parquet(os.path.join(ROOT, "data", "processed", "uplift.parquet"))
    playbook = json.load(open(os.path.join(ROOT, "data", "playbook.json"), encoding="utf-8"))
    per = args.n // len(TYPES)
    sample = pd.concat([df[df.CustomerType == t].sample(per, random_state=0) for t in TYPES]).reset_index(drop=True)

    # Scenario set: written every run so it can be inspected / human-labelled without any LLM call.
    scen = sample[["CustomerID", "Segment", "CustomerType", "ChurnProbability", "UpliftScore", "NetROI", "RiskTier"]].copy()
    scen["top_shap_driver"] = [f"{top_driver(r)[0]} ({top_driver(r)[1]:+.3f})" for _, r in sample.iterrows()]
    scen["rule_based_output"] = [json.dumps(rule_policy(r, playbook)) for _, r in sample.iterrows()]
    scen.to_csv(os.path.join(os.path.dirname(__file__), "agent_scenarios.csv"), index=False)
    if args.scenarios_only:
        print(f"wrote {len(scen)} scenarios to eval_sop/agent_scenarios.csv (no LLM called)")
        return

    if args.limit:
        sample = sample.iloc[: args.limit]

    # Wire the project's agent to the local model, unchanged otherwise.
    agent_loop.Groq = OllamaAsGroq
    orig_loop = agent_loop.run_agentic_loop
    agent_loop.run_agentic_loop = lambda m, d, p, k: orig_loop(m, d, p, k, model=args.model)
    agent_loop.at.execute_tool = leak_free_execute_tool(agent_tools.execute_tool.__wrapped__ if hasattr(agent_tools.execute_tool, "__wrapped__") else agent_tools.execute_tool)

    rows, raw_log = [], []
    ckpt = os.path.join(OUT, f"agent_eval_checkpoint_{args.model.replace(':', '_')}.jsonl")
    done = {}
    if os.path.exists(ckpt):  # resume: one line per (seed, customer) with all three approaches
        for line in open(ckpt, encoding="utf-8"):
            rec = json.loads(line)
            done[(rec["seed"], rec["customer_id"])] = rec
    t0 = time.time()
    for seed in args.seeds:
        OllamaAsGroq.seed = seed
        for i, r in sample.iterrows():
            row = r.to_dict()
            if (seed, row["CustomerID"]) in done:
                rec = done[(seed, row["CustomerID"])]
                rows += rec["rows"]
                raw_log += rec["raw"]
                continue
            n_rows0, n_raw0 = len(rows), len(raw_log)
            # agent
            ts = time.time()
            reset_budget()
            a = agent_loop.generate_retention_action_agentic(row, df, playbook, api_key="local")
            b_agent = dict(BUDGET)
            lat = time.time() - ts
            trace = a.pop("trace", [])
            out = {k: v for k, v in a.items() if k not in ("customer_id", "segment", "churn_probability", "uplift_score", "net_roi")}
            if out.get("error") is None:
                out.pop("error", None)
            s = score(row, out, trace, overflow=b_agent["overflow"])
            infra = is_infra(out.get("error", ""))
            rows.append({"approach": "agent", "seed": seed, "customer_id": row["CustomerID"], "customer_type": row["CustomerType"], "latency_s": lat, "infra_error": infra,
                         "llm_calls": b_agent["calls"], "max_est_prompt_tokens": b_agent["max_est_prompt_tokens"], **s})
            raw_log.append({"approach": "agent", "seed": seed, "customer_id": row["CustomerID"], "output": out, "trace_tools": [(t["tool"], t["args"]) for t in trace]})
            # no tools
            ts = time.time()
            reset_budget()
            try:
                raw = no_tools(row, args.model, seed)
                infra = False
            except Exception as e:  # local server overloaded / timed out: infrastructure, not model, failure
                raw, infra = f"INFRA_ERROR: {e}", True
            lat = time.time() - ts
            out = parse_json(raw) or {"raw": raw[:500]}
            s = score(row, out, overflow=BUDGET["overflow"])
            rows.append({"approach": "no_tools_llm", "seed": seed, "customer_id": row["CustomerID"], "customer_type": row["CustomerType"], "latency_s": lat, "infra_error": infra,
                         "llm_calls": BUDGET["calls"], "max_est_prompt_tokens": BUDGET["max_est_prompt_tokens"], **s})
            raw_log.append({"approach": "no_tools_llm", "seed": seed, "customer_id": row["CustomerID"], "output": out})
            # rule
            out = rule_policy(row, playbook)
            s = score(row, out)
            rows.append({"approach": "rule_based", "seed": seed, "customer_id": row["CustomerID"], "customer_type": row["CustomerType"], "latency_s": 0.0, "infra_error": False, "llm_calls": 0, "max_est_prompt_tokens": 0, **s})
            raw_log.append({"approach": "rule_based", "seed": seed, "customer_id": row["CustomerID"], "output": out})
            with open(ckpt, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"seed": seed, "customer_id": row["CustomerID"], "rows": rows[n_rows0:], "raw": raw_log[n_raw0:]}, default=str) + "\n")
            print(f"seed {seed} customer {i+1}/{len(sample)} ({time.time()-t0:.0f}s)", flush=True)

    try:
        unload(args.model)  # keep_alive: 0
        print("model unloaded (keep_alive=0)")
    except Exception as e:
        print("unload failed:", e)
    with open(os.path.join(OUT, "agent_eval_token_check.json"), "w") as fh:
        json.dump({"chars_per_token_assumed": CHARS_PER_TOKEN_EST, "num_ctx": NUM_CTX, "cold_calls": BUDGET["cold_calls"]}, fh, indent=2)
    res = pd.DataFrame(rows)
    n_infra = int(res["infra_error"].sum())
    print(f"infra errors excluded from scoring: {n_infra}")
    res_all = res
    res = res[~res["infra_error"]]
    res_all.to_csv(os.path.join(OUT, "agent_eval_rows_including_infra_errors.csv"), index=False)
    res.to_csv(os.path.join(OUT, "agent_eval_rows.csv"), index=False)
    with open(os.path.join(OUT, "agent_eval_raw_outputs.jsonl"), "w", encoding="utf-8") as fh:
        for r in raw_log:
            fh.write(json.dumps(r, default=str) + "\n")

    metrics = ["valid_json", "budget_overflow", "agrees_with_rule", "intervened_on_lost_cause_or_sleeping_dog", "cites_true_top_shap_driver",
               "cost_exceeds_model_expected_value", "playbook_query_matches_top_driver", "roi_tool_used_true_uplift", "intervene", "model_estimated_net_value_usd", "n_tool_calls", "latency_s"]
    per_seed = res.groupby(["approach", "seed"])[metrics].mean()
    summ = per_seed.groupby("approach").agg(["mean", "std"])
    # bootstrap over customers (seed-averaged per customer)
    rng = np.random.default_rng(0)
    ci = []
    for ap_name, g in res.groupby("approach"):
        cust = g.groupby("customer_id")[metrics].mean()
        rec = {"approach": ap_name, "n_customers": len(cust), "n_runs": len(g)}
        for m in metrics:
            v = cust[m].astype(float).values
            b = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(2000)]
            rec[m] = v.mean()
            rec[m + "_lo"], rec[m + "_hi"] = np.percentile(b, [2.5, 97.5])
        ci.append(rec)
    pd.DataFrame(ci).to_csv(os.path.join(OUT, "agent_eval_summary_ci.csv"), index=False)
    summ.to_csv(os.path.join(OUT, "agent_eval_summary_by_seed.csv"))
    print(pd.DataFrame(ci).round(3).T.to_string())

    # Human-validation sheet (seed = first seed)
    hv = []
    first = args.seeds[0]
    lookup = {(r["approach"], r["seed"], r["customer_id"]): r["output"] for r in raw_log}
    for _, r in sample.iterrows():
        feat, val = top_driver(r)
        for apn in ["agent", "no_tools_llm", "rule_based"]:
            o = lookup[(apn, first, r["CustomerID"])]
            hv.append({"customer_id": r["CustomerID"], "approach": apn, "customer_type": r["CustomerType"],
                       "segment": r["Segment"], "churn_probability": round(r["ChurnProbability"], 3),
                       "uplift_score": round(r["UpliftScore"], 4), "net_roi": round(r["NetROI"], 2),
                       "top_shap_driver": f"{feat} ({val:+.3f})", "output_json": json.dumps(o, default=str)[:1500],
                       "HUMAN_decision_reasonable(Y/N)": "", "HUMAN_grounded_in_data(Y/N)": "", "HUMAN_notes": ""})
    pd.DataFrame(hv).to_csv(os.path.join(OUT, "agent_eval_human_validation.csv"), index=False)
    with open(os.path.join(OUT, "agent_eval_run_info.json"), "w") as fh:
        json.dump({"model": args.model, "endpoint": "ollama local native /api/chat, num_ctx=8192, sequential", "seeds": args.seeds,
                   "temperature": OllamaAsGroq.temperature, "n_customers": len(sample),
                   "sample": "10 per CustomerType from data/processed/uplift.parquet, pandas sample random_state=0",
                   "runtime_s": round(time.time() - t0, 1), "n_infra_errors_excluded": n_infra}, fh, indent=2)


if __name__ == "__main__":
    main()
