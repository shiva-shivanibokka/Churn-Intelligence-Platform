"""
LLM Retention Action Generator
================================
Converts SHAP-derived risk factors, uplift scores, and segment profiles
into structured, personalized retention strategies via an LLM.

Uses Groq (free tier) with Llama 3.3 70B — fast inference, no cost,
and strong structured JSON output. Get a free key at console.groq.com.

This mirrors how Salesforce Einstein and HubSpot use LLMs to generate
CSM (Customer Success Manager) playbooks: the model doesn't just say
"this customer will churn" — it says "here's specifically what to do,
through which channel, with what message, at what cost, and why."

Architecture:
- Input: top SHAP features (WHY they're at risk), segment profile (WHO they are),
  churn probability, uplift score, CLV estimate, customer type classification
- Output: structured retention action with intervention type, channel,
  timing, message framing, estimated cost, and expected ROI reasoning

The prompt engineering pattern follows the "chain of thought + structured output"
approach used in enterprise AI products — forcing the LLM to reason through
the customer profile before generating the recommendation.
"""

import json
import logging
import time

from groq import Groq, RateLimitError

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_RETRY_DELAY_S = 2.0


def build_retention_prompt(
    customer_row: dict,
    segment_profile: dict,
    top_shap: dict,
    avg_clv: float = 500.0,
) -> str:
    """
    Build a structured prompt that forces Claude to reason through the
    customer's risk profile before generating the intervention recommendation.

    This prompt structure mirrors Salesforce's Einstein Copilot prompting
    for CSM playbook generation.
    """
    segment = customer_row.get("Segment", "Unknown")
    churn_prob = customer_row.get("ChurnProbability", 0)
    uplift = customer_row.get("UpliftScore", 0)
    risk_tier = customer_row.get("RiskTier", "Unknown")
    customer_type = customer_row.get("CustomerType", "Unknown")
    net_roi = customer_row.get("NetROI", 0)
    tenure = customer_row.get("Tenure", 0)
    satisfaction = customer_row.get("SatisfactionScore", 3)
    complain = customer_row.get("Complain", 0)
    hours_on_app = customer_row.get("HourSpendOnApp", 0)
    days_since_order = customer_row.get("DaySinceLastOrder", 0)
    order_count = customer_row.get("OrderCount", 0)
    cashback = customer_row.get("CashbackAmount", 0)
    coupon_used = customer_row.get("CouponUsed", 0)

    # Format SHAP features for readability
    shap_str = "\n".join(
        [
            f"  - {feat}: {'+' if val > 0 else ''}{val:.3f} ({'increases' if val > 0 else 'decreases'} churn risk)"
            for feat, val in sorted(
                top_shap.items(), key=lambda x: abs(x[1]), reverse=True
            )
        ]
    )

    # Segment profile summary
    seg_churn_rate = segment_profile.get("ChurnRate", "N/A")
    seg_engagement = segment_profile.get("EngagementScore", "N/A")
    seg_recency = segment_profile.get("RecencySignal", "N/A")
    seg_size = segment_profile.get("CustomerCount", "N/A")

    prompt = f"""You are a senior Customer Success strategist at an e-commerce company.
Your role is to analyze at-risk customer profiles and generate specific, actionable retention strategies.

## Customer Profile

**Segment:** {segment} ({seg_size} customers in this segment, {seg_churn_rate:.1%} average churn rate)
**Risk Tier:** {risk_tier} | **Churn Probability:** {churn_prob:.1%}
**Uplift Score:** {uplift:+.3f} (positive = responds to intervention, negative = does not)
**Customer Type:** {customer_type}
**Estimated Intervention ROI:** ${net_roi:.2f} (based on ${avg_clv:.0f} estimated CLV)

## Customer Behavior

- Tenure with platform: {tenure:.0f} months
- Satisfaction score: {satisfaction}/5 (1=satisfied, 5=very dissatisfied)
- Filed a complaint recently: {"Yes" if complain else "No"}
- Hours spent on app (last period): {hours_on_app:.1f} hours
- Days since last order: {days_since_order:.0f} days
- Total orders placed: {order_count:.0f}
- Cashback earned: ${cashback:.2f}
- Coupons used: {coupon_used:.0f}

## Top Risk Factors (SHAP Analysis)

The following features are the primary drivers of this customer's churn risk.
Positive SHAP values increase churn risk; negative values decrease it:

{shap_str}

## Segment Context

This customer belongs to the **{segment}** segment:
- Average engagement score: {seg_engagement:.2f}/1.0
- Average recency signal: {seg_recency:.2f} (higher = less recent purchases)
- Historical churn rate in this segment: {seg_churn_rate:.1%}

## Your Task

Based on the above profile, generate a structured retention action plan.
Think step by step:

1. What is the PRIMARY reason this customer is at risk? (based on SHAP analysis)
2. What does the Customer Type classification ({customer_type}) tell us about their receptivity to intervention?
3. What is the most appropriate intervention given their segment and risk drivers?
4. What is the optimal channel and timing?
5. What specific message framing will resonate with this customer's profile?

Output your response in the following JSON format ONLY (no other text):

{{
  "primary_risk_reason": "One sentence describing the main churn driver",
  "customer_receptivity": "Brief assessment of whether intervention will work and why",
  "intervention_type": "One of: Discount Offer / Loyalty Reward / Personalized Content / Proactive Support Call / Reactivation Campaign / Premium Upgrade / Service Recovery",
  "channel": "One of: In-App Notification / Email / Push Notification / Direct Call / SMS",
  "timing": "One of: Immediate / Within 24 hours / Within 1 week / During next session",
  "message_framing": "2-3 sentence customer-facing message (do not mention ML or churn prediction)",
  "intervention_cost_estimate": "Low ($1-5) / Medium ($10-25) / High ($50-100)",
  "expected_outcome": "Brief prediction of what happens if this intervention is executed",
  "confidence": "High / Medium / Low",
  "do_not_intervene_reason": null
}}

If the Customer Type is 'Lost Cause' or 'Sleeping Dog', set all fields to null and populate 'do_not_intervene_reason' explaining why intervention would be counterproductive.
"""
    return prompt


def generate_retention_action(
    customer_row: dict,
    segment_profile: dict,
    top_shap: dict,
    api_key: str,
    avg_clv: float = 500.0,
    model: str = "llama-3.3-70b-versatile",
) -> dict:
    """
    Generate a retention action for a single customer using Groq (free tier).

    Uses llama-3.3-70b-versatile — Groq's free tier supports ~30 RPM which
    is sufficient for batch retention action generation. In a production system
    this would be batched via async workers and cached by segment to reduce calls.
    """
    client = Groq(api_key=api_key)
    prompt = build_retention_prompt(customer_row, segment_profile, top_shap, avg_clv)
    customer_id = customer_row.get("CustomerID", "N/A")
    raw_text = "No response"

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                max_tokens=600,
                messages=[{"role": "user", "content": prompt}],
            )
            raw_text = response.choices[0].message.content.strip()

            # Strip markdown code fences if present
            if "```json" in raw_text:
                raw_text = raw_text.split("```json")[1].split("```")[0].strip()
            elif "```" in raw_text:
                raw_text = raw_text.split("```")[1].split("```")[0].strip()

            action = json.loads(raw_text)
            action["customer_id"] = customer_id
            action["segment"] = customer_row.get("Segment", "Unknown")
            action["churn_probability"] = customer_row.get("ChurnProbability", 0)
            action["uplift_score"] = customer_row.get("UpliftScore", 0)
            action["net_roi"] = customer_row.get("NetROI", 0)
            action["error"] = None
            return action

        except json.JSONDecodeError as e:
            logger.warning("JSON parse error (attempt %d/%d) for customer %s: %s", attempt, _MAX_RETRIES, customer_id, e)
            if attempt == _MAX_RETRIES:
                logger.error("All retries exhausted for customer %s.", customer_id)
                return {"customer_id": customer_id, "error": f"JSON parse error: {e}", "raw_response": raw_text}

        except RateLimitError:
            logger.warning("Rate limited (attempt %d/%d) — backing off %.1fs...", attempt, _MAX_RETRIES, _RETRY_DELAY_S * attempt)
            time.sleep(_RETRY_DELAY_S * attempt)

        except Exception as e:
            logger.error("Retention action generation failed for customer %s: %s", customer_id, e)
            return {"customer_id": customer_id, "error": str(e)}

    return {"customer_id": customer_id, "error": "Max retries exceeded."}


def generate_batch_retention_actions(
    df_uplift,
    segment_profiles,
    api_key: str,
    top_n: int = 20,
    customer_types: list | None = None,
    avg_clv: float = 500.0,
) -> list[dict]:
    """
    Generate retention actions for the top N at-risk customers.

    Filters to Persuadables by default (the only customers where
    intervention has positive expected ROI). This mirrors the production
    pattern at Salesforce and Netflix where the decision layer only
    passes actionable customers downstream to the intervention system.
    """

    if customer_types is None:
        customer_types = ["Persuadable"]

    # Filter to actionable customer types, ranked by ROI
    filtered = df_uplift[df_uplift["CustomerType"].isin(customer_types)].copy()

    if "InterventionPriority" in filtered.columns:
        filtered = filtered.sort_values("InterventionPriority").head(top_n)
    else:
        filtered = filtered.nlargest(top_n, "NetROI")

    actions = []
    for i, (idx, row) in enumerate(filtered.iterrows()):
        segment = row.get("Segment", "Unknown")

        # Get segment profile
        if segment in segment_profiles.index:
            seg_profile = segment_profiles.loc[segment].to_dict()
        else:
            seg_profile = {}

        # Parse SHAP features
        shap_raw = row.get("TopSHAPFeatures", "{}")
        try:
            top_shap = json.loads(shap_raw) if isinstance(shap_raw, str) else {}
        except Exception:
            top_shap = {}

        logger.info(
            "Generating action %d/%d for customer %s (segment: %s, churn: %.1f%%)...",
            i + 1, min(top_n, len(filtered)), row.get("CustomerID", idx), segment,
            row.get("ChurnProbability", 0) * 100,
        )

        action = generate_retention_action(
            customer_row=row.to_dict(),
            segment_profile=seg_profile,
            top_shap=top_shap,
            api_key=api_key,
            avg_clv=avg_clv,
        )
        actions.append(action)

    return actions


def format_action_for_display(action: dict) -> str:
    """Format a retention action for display in the Streamlit UI."""
    if action.get("error"):
        return f"Error generating action: {action['error']}"

    if action.get("do_not_intervene_reason"):
        return f"**No Intervention Recommended**\n\n{action['do_not_intervene_reason']}"

    lines = [
        f"**Intervention:** {action.get('intervention_type', 'N/A')}",
        f"**Channel:** {action.get('channel', 'N/A')} | **Timing:** {action.get('timing', 'N/A')}",
        f"**Cost:** {action.get('intervention_cost_estimate', 'N/A')} | **Confidence:** {action.get('confidence', 'N/A')}",
        "",
        f"**Why they're at risk:** {action.get('primary_risk_reason', 'N/A')}",
        f"**Will they respond?** {action.get('customer_receptivity', 'N/A')}",
        "",
        "**Suggested Message:**",
        f"> {action.get('message_framing', 'N/A')}",
        "",
        f"**Expected Outcome:** {action.get('expected_outcome', 'N/A')}",
    ]
    return "\n".join(lines)
