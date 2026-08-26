"""
Agent Tool Definitions and Implementations
===========================================
Defines the 6 tools exposed to the Groq tool-calling agent.

Each tool has:
  - A JSON schema (TOOL_SCHEMAS) used by the Groq API to tell the model what tools exist
  - An implementation function that queries the in-memory DataFrame

All implementations accept the DataFrame and playbook as parameters rather than
global state — this makes them testable and avoids Streamlit cache coupling.
"""

import json
import logging

logger = logging.getLogger(__name__)

# ─── Groq Tool Schemas ────────────────────────────────────────────────────────

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_customer_details",
            "description": (
                "Look up a specific customer's full behavioral profile. "
                "Returns tenure, order history, app engagement, satisfaction score, "
                "complaints, churn probability, uplift score, segment, and customer type. "
                "Use this as the first tool when asked about a specific customer."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {
                        "type": "string",
                        "description": "The CustomerID to look up (e.g., '50001')",
                    }
                },
                "required": ["customer_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_segment_benchmark",
            "description": (
                "Get average behavioral metrics for a customer segment. "
                "Use this to benchmark a specific customer against their cohort — "
                "e.g., is this customer's engagement score below the segment average?"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "segment": {
                        "type": "string",
                        "description": "Segment name: 'Champions', 'Loyal Customers', 'At-Risk', 'Price Sensitive', or 'Lapsed'",
                    }
                },
                "required": ["segment"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_top_churn_drivers",
            "description": (
                "Get the top SHAP-derived churn risk factors for a specific customer. "
                "These are the primary features driving the model's churn prediction for this customer. "
                "Always call this before recommending an intervention — the intervention should target the primary driver."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {
                        "type": "string",
                        "description": "The CustomerID to get churn drivers for",
                    }
                },
                "required": ["customer_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_intervention_roi",
            "description": (
                "Calculate the expected net ROI of a retention intervention for a customer, "
                "given their uplift score, estimated customer lifetime value (CLV), and the "
                "cost of the intervention. Always calculate ROI before recommending an action "
                "to avoid spending more than the expected retained value."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "uplift_score": {
                        "type": "number",
                        "description": "The customer's uplift score (positive = responds to intervention, negative = do not intervene)",
                    },
                    "clv": {
                        "type": "number",
                        "description": "Estimated customer lifetime value in dollars (use 500 as default if unknown)",
                    },
                    "intervention_cost": {
                        "type": "number",
                        "description": "Cost of the planned intervention in dollars (e.g., 2 for email, 15 for voucher, 75 for direct call)",
                    },
                },
                "required": ["uplift_score", "clv", "intervention_cost"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_retention_playbook",
            "description": (
                "Look up the recommended intervention type, channel, timing, and message template "
                "for a given churn risk factor. Call this after identifying the primary risk driver "
                "to find the historically best-performing intervention approach for that pattern. "
                "This is equivalent to Salesforce's Next Best Action engine."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "risk_factor": {
                        "type": "string",
                        "description": (
                            "The primary risk factor including its direction, e.g.: "
                            "'high SupportRiskScore', 'low EngagementScore', "
                            "'high RecencySignal', 'low Tenure', 'high Complain', "
                            "'high DiscountSensitivity', 'low CashbackAmount'"
                        ),
                    }
                },
                "required": ["risk_factor"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_customers",
            "description": (
                "List customers filtered by segment, risk tier, or customer type, "
                "ranked by churn probability. Use this to answer questions like "
                "'which customers in At-Risk are most urgent?' or "
                "'show me all Persuadables with the highest ROI'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "segment": {
                        "type": "string",
                        "description": "Filter by segment: 'Champions', 'Loyal Customers', 'At-Risk', 'Price Sensitive', 'Lapsed' (optional)",
                    },
                    "risk_tier": {
                        "type": "string",
                        "description": "Filter by risk tier: 'High Risk', 'Medium Risk', 'Low Risk' (optional)",
                    },
                    "customer_type": {
                        "type": "string",
                        "description": "Filter by type: 'Persuadable', 'Sure Thing', 'Lost Cause', 'Sleeping Dog' (optional)",
                    },
                    "top_n": {
                        "type": "integer",
                        "description": "Number of customers to return (default 10, max 20)",
                    },
                },
                "required": [],
            },
        },
    },
]


# ─── Tool Implementations ─────────────────────────────────────────────────────

def lookup_customer_details(df, customer_id: str) -> dict:
    row = df[df["CustomerID"].astype(str) == str(customer_id)]
    if row.empty:
        return {"error": f"Customer {customer_id} not found. Valid IDs are integers like 50001."}
    r = row.iloc[0]
    return {
        "customer_id": str(customer_id),
        "segment": str(r.get("Segment", "Unknown")),
        "risk_tier": str(r.get("RiskTier", "Unknown")),
        "customer_type": str(r.get("CustomerType", "Unknown")),
        "churn_probability": round(float(r.get("ChurnProbability", 0)), 3),
        "uplift_score": round(float(r.get("UpliftScore", 0)), 3),
        "net_roi_at_500_clv": round(float(r.get("NetROI", 0)), 2),
        "tenure_months": int(r.get("Tenure", 0)),
        "satisfaction_score_1to5": int(r.get("SatisfactionScore", 3)),
        "satisfaction_note": "1=satisfied, 5=very dissatisfied",
        "filed_complaint_recently": bool(r.get("Complain", 0)),
        "hours_on_app_last_period": round(float(r.get("HourSpendOnApp", 0)), 1),
        "days_since_last_order": int(r.get("DaySinceLastOrder", 0)),
        "total_orders": int(r.get("OrderCount", 0)),
        "cashback_earned_usd": round(float(r.get("CashbackAmount", 0)), 2),
        "coupons_used": int(r.get("CouponUsed", 0)),
        "actually_churned": bool(r.get("Churn", 0)),
    }


def get_segment_benchmark(df, segment: str) -> dict:
    seg_df = df[df["Segment"] == segment]
    if seg_df.empty:
        valid = df["Segment"].unique().tolist()
        return {"error": f"Segment '{segment}' not found.", "valid_segments": valid}
    return {
        "segment": segment,
        "customer_count": int(len(seg_df)),
        "actual_churn_rate": round(float(seg_df["Churn"].mean()), 3),
        "avg_predicted_churn_probability": round(float(seg_df["ChurnProbability"].mean()), 3),
        "avg_uplift_score": round(float(seg_df["UpliftScore"].mean()), 3),
        "pct_high_risk": round(float((seg_df["RiskTier"] == "High Risk").mean()), 3),
        "pct_persuadable": round(float((seg_df["CustomerType"] == "Persuadable").mean()), 3),
        "avg_tenure_months": round(float(seg_df["Tenure"].mean()), 1) if "Tenure" in seg_df else None,
        "avg_satisfaction_score": round(float(seg_df["SatisfactionScore"].mean()), 2) if "SatisfactionScore" in seg_df else None,
        "avg_days_since_last_order": round(float(seg_df["DaySinceLastOrder"].mean()), 1) if "DaySinceLastOrder" in seg_df else None,
        "avg_hours_on_app": round(float(seg_df["HourSpendOnApp"].mean()), 1) if "HourSpendOnApp" in seg_df else None,
        "complaint_rate": round(float(seg_df["Complain"].mean()), 3) if "Complain" in seg_df else None,
    }


def get_top_churn_drivers(df, customer_id: str) -> dict:
    row = df[df["CustomerID"].astype(str) == str(customer_id)]
    if row.empty:
        return {"error": f"Customer {customer_id} not found."}
    r = row.iloc[0]
    shap_raw = r.get("TopSHAPFeatures", "{}")
    try:
        shap_dict = json.loads(shap_raw) if isinstance(shap_raw, str) else {}
    except Exception:
        shap_dict = {}

    sorted_shap = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)
    return {
        "customer_id": str(customer_id),
        "primary_driver": sorted_shap[0][0] if sorted_shap else "Unknown",
        "top_risk_factors": [
            {
                "feature": feat,
                "shap_value": round(val, 4),
                "effect": "increases churn risk" if val > 0 else "decreases churn risk",
                "magnitude": "strong" if abs(val) > 0.1 else "moderate" if abs(val) > 0.05 else "weak",
            }
            for feat, val in sorted_shap[:6]
        ],
        "interpretation": (
            f"The strongest churn driver for this customer is '{sorted_shap[0][0]}' "
            f"(SHAP: {sorted_shap[0][1]:+.3f}). "
            f"This {'pushes toward churn' if sorted_shap[0][1] > 0 else 'reduces churn risk'}."
        ) if sorted_shap else "No SHAP data available.",
    }


def calculate_intervention_roi(uplift_score: float, clv: float, intervention_cost: float) -> dict:
    expected_value = float(uplift_score) * float(clv)
    net_roi = expected_value - float(intervention_cost)
    roi_pct = (net_roi / intervention_cost * 100) if intervention_cost > 0 else 0
    break_even_uplift = intervention_cost / clv if clv > 0 else 0
    return {
        "uplift_score": round(uplift_score, 3),
        "clv_assumption": round(clv, 2),
        "intervention_cost": round(intervention_cost, 2),
        "expected_retained_value": round(expected_value, 2),
        "net_roi": round(net_roi, 2),
        "roi_percentage": round(roi_pct, 1),
        "break_even_uplift_needed": round(break_even_uplift, 3),
        "recommendation": "Intervene — positive ROI" if net_roi > 0 else "Skip — intervention costs more than expected retained value",
        "verdict": "positive" if net_roi > 0 else "negative",
    }


def search_retention_playbook(playbook: dict, risk_factor: str) -> dict:
    normalized = risk_factor.lower().replace(" ", "_").replace("-", "_")

    # Exact match
    if normalized in playbook:
        return {**playbook[normalized], "match_type": "exact", "query": risk_factor}

    # Keyword match — find playbook entries whose key words appear in the query
    best_match = None
    best_score = 0
    for key, value in playbook.items():
        if key == "default":
            continue
        key_words = [w for w in key.split("_") if len(w) > 3]
        score = sum(1 for w in key_words if w in normalized)
        if score > best_score:
            best_score = score
            best_match = (key, value)

    if best_match and best_score > 0:
        return {**best_match[1], "match_type": "fuzzy", "matched_key": best_match[0], "query": risk_factor}

    # Fallback to default
    return {**playbook.get("default", {}), "match_type": "default", "query": risk_factor,
            "note": "No specific playbook entry found — using default recommendation."}


def list_customers(df, segment=None, risk_tier=None, customer_type=None, top_n=10) -> dict:
    filtered = df.copy()
    if segment:
        filtered = filtered[filtered["Segment"] == segment]
    if risk_tier:
        filtered = filtered[filtered["RiskTier"] == risk_tier]
    if customer_type:
        filtered = filtered[filtered["CustomerType"] == customer_type]

    top_n = min(int(top_n or 10), 20)
    result = filtered.nlargest(top_n, "ChurnProbability")

    customers = [
        {
            "customer_id": str(row.get("CustomerID", "N/A")),
            "segment": str(row.get("Segment", "Unknown")),
            "risk_tier": str(row.get("RiskTier", "Unknown")),
            "customer_type": str(row.get("CustomerType", "Unknown")),
            "churn_probability": round(float(row.get("ChurnProbability", 0)), 3),
            "uplift_score": round(float(row.get("UpliftScore", 0)), 3),
            "net_roi": round(float(row.get("NetROI", 0)), 2),
        }
        for _, row in result.iterrows()
    ]
    return {
        "total_matching": int(len(filtered)),
        "showing": len(customers),
        "filters_applied": {k: v for k, v in {"segment": segment, "risk_tier": risk_tier, "customer_type": customer_type}.items() if v},
        "customers": customers,
    }


# ─── Tool Dispatcher ──────────────────────────────────────────────────────────

def execute_tool(name: str, args: dict, df, playbook: dict) -> dict:
    """Dispatch a tool call by name and return the result as a dict."""
    try:
        if name == "lookup_customer_details":
            return lookup_customer_details(df, args["customer_id"])
        if name == "get_segment_benchmark":
            return get_segment_benchmark(df, args["segment"])
        if name == "get_top_churn_drivers":
            return get_top_churn_drivers(df, args["customer_id"])
        if name == "calculate_intervention_roi":
            return calculate_intervention_roi(
                args["uplift_score"], args["clv"], args["intervention_cost"]
            )
        if name == "search_retention_playbook":
            return search_retention_playbook(playbook, args["risk_factor"])
        if name == "list_customers":
            return list_customers(
                df,
                segment=args.get("segment"),
                risk_tier=args.get("risk_tier"),
                customer_type=args.get("customer_type"),
                top_n=args.get("top_n", 10),
            )
        return {"error": f"Unknown tool: {name}"}
    except KeyError as e:
        return {"error": f"Missing required argument for tool '{name}': {e}"}
    except Exception as e:
        logger.error("Tool '%s' failed: %s", name, e)
        return {"error": str(e)}
