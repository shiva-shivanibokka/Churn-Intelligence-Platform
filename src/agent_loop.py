"""
Agentic Tool-Calling Loop
==========================
Implements a ReAct-style (Reason + Act) loop using Groq's tool-calling API.
The agent iteratively calls tools to gather data, then produces a final answer.

Two modes:
  - BATCH: generates structured JSON retention plans (same output schema as retention_llm.py)
  - CHAT:  answers free-form questions from CSMs in natural language

Architecture mirrors what Salesforce Einstein Copilot and HubSpot AI do:
  - Agent receives a task and a set of tools
  - Agent decides what data to retrieve (not pre-stuffed into the prompt)
  - Agent's reasoning trace is logged for audit and explainability
"""

import json
import logging
import time

from groq import Groq, RateLimitError

import agent_tools as at

logger = logging.getLogger(__name__)

_MAX_ROUNDS = 5
_RATE_LIMIT_BACKOFF_S = 3.0
_MODEL = "llama-3.3-70b-versatile"

# ─── System Prompts ───────────────────────────────────────────────────────────

SYSTEM_PROMPT_BATCH = """You are a Customer Success AI generating structured retention action plans.

Use the available tools to gather all necessary data before making a recommendation.
Follow this sequence:
1. get_top_churn_drivers — identify WHY this customer is at risk
2. search_retention_playbook — find the best intervention for that risk factor
3. get_segment_benchmark — compare the customer to their segment average
4. calculate_intervention_roi — verify the intervention is financially justified
5. lookup_customer_details — get any remaining behavioral context you need

After calling all relevant tools, output your final recommendation as valid JSON ONLY (no other text):

{
  "primary_risk_reason": "One sentence describing the main churn driver from SHAP analysis",
  "customer_receptivity": "Assessment of whether intervention will work and why",
  "intervention_type": "One of: Discount Offer / Loyalty Reward / Personalized Content / Proactive Support Call / Reactivation Campaign / Premium Upgrade / Service Recovery",
  "channel": "One of: In-App Notification / Email / Push Notification / Direct Call / SMS",
  "timing": "One of: Immediate / Within 24 hours / Within 1 week / During next session",
  "message_framing": "2-3 sentence customer-facing message (do not mention ML, churn, or scores)",
  "intervention_cost_estimate": "Low ($1-5) / Medium ($10-25) / High ($50-100)",
  "expected_outcome": "Brief prediction of result if intervention is executed",
  "confidence": "High / Medium / Low"
}

If the customer type is Lost Cause or Sleeping Dog, output:
{"do_not_intervene_reason": "explanation of why intervention would be counterproductive"}"""


SYSTEM_PROMPT_CHAT = """You are an AI Customer Success Assistant for an e-commerce platform.

You help Customer Success Managers (CSMs) understand their customers and decide what actions to take.
You have access to tools to query customer data, segment benchmarks, churn risk drivers, intervention ROI, and the retention playbook.

Guidelines:
- Always use tools to get real data — never make up numbers or customer details
- Be specific and cite the data you retrieved
- When recommending an intervention, always check the playbook and calculate ROI first
- Keep responses concise and actionable — a CSM needs to act, not read a report
- When listing customers, show the most critical ones first
- If asked about a customer ID you can't find, say so clearly

You can answer questions like:
- "Tell me about customer 50123 before I call them"
- "Which customers in At-Risk are most urgent right now?"
- "Is it worth offering a discount to customer 50456?"
- "How does customer 50789 compare to their segment?"
- "List the top 5 Persuadables with the highest ROI"
- "What should I say to a customer who just complained?" """


# ─── Core Loop ────────────────────────────────────────────────────────────────

def run_agentic_loop(
    messages: list,
    df,
    playbook: dict,
    api_key: str,
    model: str = _MODEL,
    max_rounds: int = _MAX_ROUNDS,
) -> dict:
    """
    Run a tool-calling agent loop.

    Args:
        messages:   Initial message list (system + user messages).
        df:         Customer DataFrame for tool queries.
        playbook:   Loaded playbook dict from data/playbook.json.
        api_key:    Groq API key.
        model:      Groq model name.
        max_rounds: Maximum tool-calling rounds before forcing a final answer.

    Returns:
        {
            "response": str,       final answer text
            "trace": list[dict],   tool calls and results per round
            "messages": list,      full message history (for multi-turn continuation)
            "error": str | None,
        }
    """
    client = Groq(api_key=api_key)
    trace = []
    api_messages = list(messages)

    for round_num in range(1, max_rounds + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=api_messages,
                tools=at.TOOL_SCHEMAS,
                tool_choice="auto",
                max_tokens=1500,
            )
        except RateLimitError:
            logger.warning("Rate limited on round %d — backing off %.1fs", round_num, _RATE_LIMIT_BACKOFF_S)
            time.sleep(_RATE_LIMIT_BACKOFF_S)
            continue
        except Exception as e:
            logger.error("API error on round %d: %s", round_num, e)
            return {"response": None, "trace": trace, "messages": api_messages, "error": str(e)}

        msg = response.choices[0].message
        finish_reason = response.choices[0].finish_reason

        # Build the assistant message dict to append
        assistant_msg: dict = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ]

        api_messages.append(assistant_msg)

        # No tool calls → final answer
        if not msg.tool_calls or finish_reason == "stop":
            return {
                "response": msg.content or "",
                "trace": trace,
                "messages": api_messages,
                "error": None,
            }

        # Execute each tool call and collect results
        for tc in msg.tool_calls:
            tool_name = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}

            logger.debug("Round %d: calling tool '%s' with args %s", round_num, tool_name, args)
            result = at.execute_tool(tool_name, args, df, playbook)
            result_str = json.dumps(result, default=str)

            trace.append({
                "round": round_num,
                "tool": tool_name,
                "args": args,
                "result": result,
            })

            api_messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result_str,
            })

    # Max rounds hit — ask for final answer without tools
    logger.warning("Max rounds (%d) reached — requesting final answer without tools.", max_rounds)
    try:
        final = client.chat.completions.create(
            model=model,
            messages=api_messages + [{"role": "user", "content": "Summarise your findings and give a final recommendation now."}],
            max_tokens=800,
        )
        return {
            "response": final.choices[0].message.content or "Max rounds reached without a final answer.",
            "trace": trace,
            "messages": api_messages,
            "error": None,
        }
    except Exception as e:
        return {"response": None, "trace": trace, "messages": api_messages, "error": str(e)}


# ─── Batch Agentic Generation ─────────────────────────────────────────────────

def generate_retention_action_agentic(
    customer_row: dict,
    df,
    playbook: dict,
    api_key: str,
    avg_clv: float = 500.0,
) -> dict:
    """
    Agentic replacement for retention_llm.generate_retention_action.
    Uses tool calling instead of a single pre-stuffed prompt.
    Returns the same schema as the non-agentic version, plus a 'trace' key.
    """
    customer_id = str(customer_row.get("CustomerID", "N/A"))
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_BATCH},
        {
            "role": "user",
            "content": (
                f"Generate a retention action plan for customer {customer_id}. "
                f"Their segment is '{customer_row.get('Segment', 'Unknown')}', "
                f"churn probability is {customer_row.get('ChurnProbability', 0):.1%}, "
                f"uplift score is {customer_row.get('UpliftScore', 0):+.3f}, "
                f"and customer type is '{customer_row.get('CustomerType', 'Unknown')}'. "
                f"Assume CLV = ${avg_clv:.0f}. Use tools to gather all supporting data first."
            ),
        },
    ]

    result = run_agentic_loop(messages, df, playbook, api_key)

    if result["error"]:
        return {"customer_id": customer_id, "error": result["error"], "trace": result["trace"]}

    raw = result["response"] or ""
    try:
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0].strip()
        action = json.loads(raw)
    except json.JSONDecodeError:
        action = {"error": "Could not parse JSON from agent response.", "raw_response": raw[:500]}

    action["customer_id"] = customer_id
    action["segment"] = customer_row.get("Segment", "Unknown")
    action["churn_probability"] = float(customer_row.get("ChurnProbability", 0))
    action["uplift_score"] = float(customer_row.get("UpliftScore", 0))
    action["net_roi"] = float(customer_row.get("NetROI", 0))
    action["trace"] = result["trace"]
    action.setdefault("error", None)
    return action


def generate_batch_agentic(
    df_uplift,
    df_full,
    playbook: dict,
    segment_profiles,
    api_key: str,
    top_n: int = 5,
    avg_clv: float = 500.0,
) -> list:
    """Generate retention actions for top N Persuadables using the agentic loop."""
    filtered = df_uplift[df_uplift["CustomerType"] == "Persuadable"].copy()
    if "InterventionPriority" in filtered.columns:
        filtered = filtered.sort_values("InterventionPriority").head(top_n)
    else:
        filtered = filtered.nlargest(top_n, "NetROI")

    actions = []
    for i, (_, row) in enumerate(filtered.iterrows()):
        logger.info(
            "Agentic generation %d/%d — customer %s (%s, churn %.1f%%)",
            i + 1, len(filtered),
            row.get("CustomerID", "N/A"),
            row.get("Segment", "?"),
            row.get("ChurnProbability", 0) * 100,
        )
        action = generate_retention_action_agentic(
            customer_row=row.to_dict(),
            df=df_full,
            playbook=playbook,
            api_key=api_key,
            avg_clv=avg_clv,
        )
        actions.append(action)

    return actions
