"""
Record real agent runs, so the demo works for someone with no API key.

Why
---
The agent route is bring-your-own-key: this deployment holds no Groq
credential, because a shared free tier is 100,000 tokens a day for every
visitor combined and the first person to hold down the button turns the feature
off for everyone else.

That is the right trade for anyone willing to paste a key. It is the wrong one
for someone evaluating the project in three minutes, who will not sign up for a
Groq account to see whether the agent works — and "add your own key" is
indistinguishable, at a glance, from "this does not run".

So the runs below are real, recorded once against the live Supabase data with a
real key, and committed as static JSON. A visitor with no key watches an actual
conversation with its actual tool calls; a visitor with a key runs their own.
Neither is a mock.

What it drives
--------------
The real `/api/agent` route, over HTTP, against a locally running dashboard —
not the Python agent in `src/agent_loop.py`. Those are two different
implementations: the Python one is the Streamlit prototype's, with six tools
over a dataframe, and the deployed one has twelve over Supabase. Recording the
prototype would produce a replay that does not match the agent a visitor with a
key actually runs, which is the one thing a replay must not do.

Driving the route also exercises the bring-your-own-key header path end to end,
so the recording doubles as a test of it.

Usage
-----
    npm --prefix dashboard run dev                 # in another terminal
    python scripts/record_agent_runs.py            # record everything
    python scripts/record_agent_runs.py --check    # verify the committed files

Requires GROQ_API_KEY in .env. Costs a handful of free-tier requests.

The key
-------
It is read from .env, never printed, and every recorded byte is scrubbed before
it is written. The scrubber matches the key's actual value and Groq's `gsk_`
shape — not the substring "key", which in an earlier project redacted
`input_tokens` because it contained "token" and destroyed a run mid-flight.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

OUT_DIR = ROOT / "dashboard" / "public" / "agent-runs"

# The route's own configuration, restated here so a recording is made with the
# same model and round limit the live agent uses.
# Which provider and model the recordings are made with, and where each
# provider's key lives in .env.
#
# Groq is the default because its free tier costs nothing, which matters for a
# script that will be re-run whenever the recordings need refreshing. Any of the
# providers the dashboard supports works — they share one OpenAI-compatible code
# path — so `--provider anthropic` records with Claude instead.
#
# The model previously named here was retired by Groq mid-project, which is
# exactly the drift a recording should not silently preserve; `--model` overrides
# it and the dashboard itself no longer hardcodes one at all.
PROVIDER_KEYS = {
    "groq": ("GROQ_API_KEY", "openai/gpt-oss-120b"),
    "openai": ("OPENAI_API_KEY", "gpt-4o-mini"),
    "anthropic": ("ANTHROPIC_API_KEY", "claude-sonnet-5"),
    "gemini": ("GEMINI_API_KEY", "gemini-2.0-flash"),
    "openrouter": ("OPENROUTER_API_KEY", "openai/gpt-4o-mini"),
    "cerebras": ("CEREBRAS_API_KEY", "llama-3.3-70b"),
}

# The route's own limits, restated so a recording is made under the same ones
# the live agent uses.
MAX_ROUNDS = 5
ANSWER_MAX_TOKENS = 4096

# Recording is a batch of large requests, which is exactly what a
# tokens-per-minute limit is designed to slow down. Pace between runs and wait
# it out rather than giving up on the recording.
PAUSE_BETWEEN_RUNS_SECONDS = 25
RATE_LIMIT_BACKOFF_SECONDS = 70
MAX_RATE_LIMIT_RETRIES = 5

# Chat questions worth recording: each one exercises a different set of tools,
# so the trace shows the agent choosing rather than always doing the same thing.
CHAT_QUESTIONS = [
    "Which segment has the most revenue at risk, and how much?",
    "Compare all five segments — where should we focus retention spend first?",
    "Which persuadable customers have not been contacted yet? Give me the top five.",
]


def load_env(var_name: str) -> str:
    """
    Read one provider's key from .env, letting .env win over the environment.

    `setdefault` was wrong here, and wrong in a way that wastes money. A stale
    `ANTHROPIC_API_KEY` exported by some other tool would keep its place, the
    project's own .env would be ignored, and the run would fail on a dead
    credential while the file you just edited sat there looking correct. This
    exact thing happened: same key length, different value, and a 401 that looked
    like the new key was bad.

    A project-local .env is the authority for the project. The environment is the
    fallback, so CI and containers still work with no file present.
    """
    env_path = ROOT / ".env"
    from_file: dict[str, str] = {}
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                name, _, value = line.partition("=")
                from_file[name.strip()] = value.strip()

    key = (from_file.get(var_name) or os.environ.get(var_name, "")).strip()
    if not key:
        raise SystemExit(
            f"{var_name} is not set. Add it to .env — it is used once, here, to "
            "record the runs, and never ships with the deployment."
        )
    return key


def make_scrubber(secret: str):
    """
    Remove the recording key from anything about to be written to disk.

    Matches the literal value and Groq's key shape. Deliberately not a keyword
    match on "key" or "token": a scrubber built that way once redacted
    `input_tokens` out of a benchmark because the field name contained "token",
    which corrupted the run it was supposed to protect.
    """
    pattern = re.compile(
        "|".join(
            [
                re.escape(secret),
                r"gsk_[A-Za-z0-9]{20,}",
                r"csk-[A-Za-z0-9]{20,}",
                r"sk-ant-[A-Za-z0-9_-]{20,}",
                r"sk-or-[A-Za-z0-9_-]{20,}",
                r"sk-[A-Za-z0-9_-]{20,}",
                r"AIza[A-Za-z0-9_-]{20,}",
            ]
        )
    )

    def scrub(obj):
        if isinstance(obj, str):
            return pattern.sub("***redacted***", obj)
        if isinstance(obj, dict):
            return {k: scrub(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [scrub(v) for v in obj]
        return obj

    return scrub


def fetch_persuadables(limit: int = 3) -> list[dict]:
    """The same query the retention page runs, so the recording uses real rows."""
    import pandas as pd

    df = pd.read_parquet(ROOT / "data" / "processed" / "uplift.parquet")
    top = (
        df[df["CustomerType"] == "Persuadable"]
        .nlargest(limit, "NetROI")
        .loc[:, ["CustomerID", "Segment", "ChurnProbability", "UpliftScore", "NetROI", "CustomerType"]]
    )
    return [
        {
            "customer_id": str(r.CustomerID),
            "segment": str(r.Segment),
            "churn_probability": float(r.ChurnProbability),
            "uplift_score": float(r.UpliftScore),
            "net_roi": float(r.NetROI),
            "customer_type": str(r.CustomerType),
        }
        for r in top.itertuples()
    ]


def post_agent(
    base_url: str, key: str, payload: dict, provider: str, model: str, attempt: int = 1
) -> dict:
    """
    One call to the real route, with the key in the header the UI uses.

    Retries on 429 with a long backoff. The route deliberately surfaces a rate
    limit immediately rather than letting the SDK sit on it — which is right for
    a visitor waiting on a page, and means a batch script has to do the waiting
    itself. Groq's free tier limits tokens per minute, and a twelve-tool ReAct
    loop spends a lot of them at once.
    """
    import urllib.error  # noqa: PLC0415
    import urllib.request  # noqa: PLC0415

    # The provider is named in the body; the base URL is resolved server-side
    # from the fixed provider map, never sent from a caller.
    payload = {**payload, "provider": provider, "model": model}
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/agent",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "x-llm-key": key},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        if exc.code == 429 and attempt <= MAX_RATE_LIMIT_RETRIES:
            wait = RATE_LIMIT_BACKOFF_SECONDS * attempt
            print(f"      rate limited — waiting {wait}s (attempt {attempt})", flush=True)
            time.sleep(wait)
            return post_agent(base_url, key, payload, provider, model, attempt + 1)
        raise SystemExit(
            f"The agent route returned {exc.code}. Body: {body[:400]}\n"
            "A 401 here means the key was rejected; a 429 means the quota is spent."
        ) from exc
    except urllib.error.URLError as exc:
        raise SystemExit(
            f"Could not reach {base_url}. Start the dashboard first:\n"
            "    npm --prefix dashboard run dev"
        ) from exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify committed recordings")
    parser.add_argument(
        "--base-url",
        default="http://localhost:3000",
        help="where the dashboard is running (default: %(default)s)",
    )
    parser.add_argument(
        "--provider",
        default="groq",
        choices=sorted(PROVIDER_KEYS),
        help="which provider to record with (default: %(default)s — its free tier costs nothing)",
    )
    parser.add_argument("--model", default=None, help="override the provider's default model")
    args = parser.parse_args()

    if args.check:
        return check()

    provider = args.provider
    var_name, default_model = PROVIDER_KEYS[provider]
    model = args.model or default_model
    key = load_env(var_name)
    scrub = make_scrubber(key)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    customers = fetch_persuadables()
    print(f"Recording against {len(customers)} real Persuadables and {len(CHAT_QUESTIONS)} questions.")
    print(f"Driving {args.base_url}/api/agent with {provider} / {model}.\n")

    index = []

    for i, customer in enumerate(customers, 1):
        print(f"  [batch {i}/{len(customers)}] {customer['customer_id']} …", flush=True)
        started = time.time()
        data = post_agent(args.base_url, key, {"mode": "batch", "customer": customer}, provider, model)
        record = scrub(
            {
                "id": f"batch-{customer['customer_id']}",
                "mode": "batch",
                "provider": provider,
                "model": model,
                "max_rounds": MAX_ROUNDS,
                "answer_max_tokens": ANSWER_MAX_TOKENS,
                "customer": customer,
                "action": data.get("action"),
                "trace": data.get("trace") or (data.get("action") or {}).get("trace") or [],
                "truncated": bool(data.get("truncated")),
                "elapsed_seconds": round(time.time() - started, 2),
            }
        )
        _write(record)
        index.append(_summary(record))
        time.sleep(PAUSE_BETWEEN_RUNS_SECONDS)

    for i, question in enumerate(CHAT_QUESTIONS, 1):
        print(f"  [chat {i}/{len(CHAT_QUESTIONS)}] {question[:50]}…", flush=True)
        started = time.time()
        data = post_agent(
            args.base_url, key, {"mode": "chat", "message": question, "history": []}, provider, model
        )
        record = scrub(
            {
                "id": f"chat-{i}",
                "mode": "chat",
                "provider": provider,
                "model": model,
                "max_rounds": MAX_ROUNDS,
                "answer_max_tokens": ANSWER_MAX_TOKENS,
                "question": question,
                "response": data.get("response"),
                "trace": data.get("trace", []),
                "truncated": bool(data.get("truncated")),
                "elapsed_seconds": round(time.time() - started, 2),
            }
        )
        _write(record)
        index.append(_summary(record))
        time.sleep(PAUSE_BETWEEN_RUNS_SECONDS)

    (OUT_DIR / "index.json").write_text(
        json.dumps({"provider": provider, "model": model, "runs": index}, indent=2),
        encoding="utf-8",
    )
    print(f"\nWrote {len(index)} recordings to {OUT_DIR.relative_to(ROOT)}")
    return check()


def _write(record: dict) -> None:
    (OUT_DIR / f"{record['id']}.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )


def _summary(record: dict) -> dict:
    return {
        "id": record["id"],
        "mode": record["mode"],
        "label": record.get("question") or f"Retention plan for {record['customer']['customer_id']}",
        "tool_calls": len(record.get("trace", [])),
        "elapsed_seconds": record["elapsed_seconds"],
        "truncated": record["truncated"],
    }


def check() -> int:
    """
    Verify the committed recordings, rather than assuming they are fine.

    Checks that every run in the index exists, carries a real trace, is not
    truncated, and — the one that matters — contains no key-shaped string. A
    recording is a file that gets served to the public; it must not be able to
    ship a credential.
    """
    index_path = OUT_DIR / "index.json"
    if not index_path.exists():
        print(f"No recordings at {OUT_DIR.relative_to(ROOT)}.", file=sys.stderr)
        return 1

    index = json.loads(index_path.read_text(encoding="utf-8"))
    problems: list[str] = []

    for entry in index["runs"]:
        path = OUT_DIR / f"{entry['id']}.json"
        if not path.exists():
            problems.append(f"{entry['id']}: indexed but missing")
            continue

        raw = path.read_text(encoding="utf-8")
        if re.search(
            r"gsk_[A-Za-z0-9]{20,}|csk-[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{20,}|AIza[A-Za-z0-9_-]{20,}",
            raw,
        ):
            problems.append(f"{entry['id']}: contains an unredacted API key")

        record = json.loads(raw)
        if not record.get("trace"):
            problems.append(f"{entry['id']}: no tool calls recorded")
        if record.get("truncated"):
            problems.append(f"{entry['id']}: answer hit the token ceiling")
        if record["mode"] == "batch":
            # `action` being truthy is not the same as a plan existing. The route
            # returns {"error": ..., "raw": ..., "trace": ...} when it cannot parse
            # the model's reply, and a dict with an `error` key in it is every bit
            # as truthy as a real plan — so this check passed two recordings that
            # said "Could not parse agent response".
            action = record.get("action") or {}
            if action.get("error"):
                problems.append(f"{entry['id']}: agent errored — {action['error']}")
            else:
                missing = [
                    field
                    for field in ("intervention_type", "channel", "timing", "message_framing")
                    if not action.get(field)
                ]
                if missing and not action.get("do_not_intervene_reason"):
                    problems.append(f"{entry['id']}: plan is missing {missing}")
        if record["mode"] == "chat" and not record.get("response"):
            problems.append(f"{entry['id']}: no response produced")

    if problems:
        print("Recorded agent runs have problems:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    total_calls = sum(e["tool_calls"] for e in index["runs"])
    print(f"{len(index['runs'])} recordings OK — {total_calls} tool calls, no keys, none truncated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
