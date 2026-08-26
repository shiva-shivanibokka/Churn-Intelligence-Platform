import { NextRequest, NextResponse } from "next/server";
import GroqClient from "groq-sdk";
import type { ChatCompletionMessageParam, ChatCompletionTool } from "groq-sdk/resources/chat/completions";
import { createClient } from "@supabase/supabase-js";
import { AGENT_MODEL } from "@/lib/models";

/**
 * Bring-your-own-key, and only that.
 *
 * There is deliberately no shared key on this deployment. A public agent
 * endpoint backed by the owner's Groq free tier is a 100k-token-a-day budget
 * that any visitor — or any script — can finish, and when they do the feature
 * is dead for everyone else until the daily reset. Worse, it is dead in a way
 * that reads as "this project is broken" rather than "someone used it up".
 *
 * So the visitor brings their own credential. The agent is then never
 * rate-limited by someone else's usage, this project has no secret to leak or
 * rotate, and the demo cannot be turned off by a stranger with a for-loop.
 *
 * The key travels in a request header, constructs a client for that one
 * request, and is never written down: not to Supabase, not to a log line, not
 * into `retention_actions`, and never into a URL, where it would land in
 * Vercel's access logs and the visitor's browser history.
 *
 * The base URL is deliberately NOT configurable. Accepting a caller-supplied
 * endpoint alongside a caller-supplied key turns this route into an open proxy:
 * anything with a `fetch` could point it at an internal address and read the
 * response back through us. The host is fixed; only the credential varies.
 */

/** Groq keys look like `gsk_…`. A cheap shape check, not authentication. */
function readUserKey(req: NextRequest): string | null {
  const raw = req.headers.get("x-groq-key")?.trim();
  if (!raw) return null;
  if (!/^gsk_[A-Za-z0-9]{20,}$/.test(raw)) return null;
  return raw;
}

/**
 * Never let a key reach a log or a response body.
 *
 * Provider SDKs sometimes echo request context into error messages, and this
 * route's catch-all returns the message to the browser. One bad interaction
 * between those two would hand a visitor's credential to whoever is watching,
 * so it is scrubbed on the way out regardless of what any SDK version does.
 */
function scrub(text: string): string {
  return text.replace(/gsk_[A-Za-z0-9]{20,}/g, "gsk_***");
}

/** Groq reports "out of quota" a few different ways. */
function isQuotaError(err: unknown): boolean {
  const status = (err as { status?: number })?.status;
  if (status === 429) return true;
  const message = String((err as { message?: string })?.message ?? err).toLowerCase();
  return message.includes("rate limit") || message.includes("quota");
}

function isAuthError(err: unknown): boolean {
  const status = (err as { status?: number })?.status;
  return status === 401 || status === 403;
}

/**
 * A hosted model that has been retired.
 *
 * This is worth its own branch because it happened: Groq withdrew
 * `llama-3.3-70b-versatile` and every request began returning 404
 * `model_not_found`, which reached the browser as a generic 500. The agent
 * looked broken rather than misconfigured, and nothing in CI could notice — the
 * model name is a valid string in correct code that simply stopped existing.
 */
function isRetiredModel(err: unknown): boolean {
  const status = (err as { status?: number })?.status;
  const message = String((err as { message?: string })?.message ?? err);
  return status === 404 && message.includes("model");
}

// Read client — anon key, respects RLS
const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL || "https://placeholder.supabase.co",
  process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY || "placeholder-key"
);

// Write client — the service role key bypasses RLS for server-side inserts.
//
// It still falls back to the anon key so that a missing variable degrades to a
// read-only agent rather than a dead route, but the fallback is no longer
// silent. Every generated plan was being refused by RLS with "new row violates
// row-level security policy", logged server-side, and then reported to the
// browser as a perfectly good 200 — so the dashboard showed a plan, the user
// assumed it was recorded, and Audit & Analytics stopped gaining rows for six
// weeks without a single visible symptom.
const HAS_SERVICE_ROLE = Boolean(process.env.SUPABASE_SERVICE_ROLE_KEY);
if (!HAS_SERVICE_ROLE) {
  console.warn(
    "SUPABASE_SERVICE_ROLE_KEY is not set — retention plans will generate but " +
      "cannot be saved, because RLS blocks inserts made with the anon key."
  );
}
const supabaseAdmin = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL || "https://placeholder.supabase.co",
  process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY || "placeholder-key"
);

// Single source of truth, shared with the sidebar that credits it.
const MODEL = AGENT_MODEL;
const MAX_ROUNDS = 5;

// Shared by every call in the loop. The forced final answer used to get 1,500
// while the rounds got 4,096, so the one message that has to carry the whole
// recommendation had the tightest budget of any of them.
const ANSWER_MAX_TOKENS = 4096;

/**
 * Time budget, because the SDK's defaults turn a rate limit into a hang.
 *
 * groq-sdk retries a 429 on its own, honouring Groq's `retry-after` — which on
 * the free tier can be minutes. A single rate-limited request was observed
 * sitting for **5.6 minutes** before dying with "Connection error". On Vercel
 * that is not a slow answer, it is a function timeout at 60 seconds
 * (`maxDuration` in vercel.json) with nothing useful to show the visitor, and
 * it makes the quota branch below unreachable — the very case it exists for.
 *
 * So: one short retry for a genuine blip, a hard per-request ceiling, and a
 * deadline across the whole ReAct loop, since five rounds of 20s would blow the
 * function limit even with every individual call inside its own.
 */
const REQUEST_TIMEOUT_MS = 20_000;
const REQUEST_RETRIES = 1;
const LOOP_BUDGET_MS = 50_000;

// ─── Tool definitions ──────────────────────────────────────────────────────────

const TOOLS: ChatCompletionTool[] = [
  {
    type: "function",
    function: {
      name: "get_top_churn_drivers",
      description: "Returns the top SHAP-based churn drivers for a specific customer.",
      parameters: {
        type: "object",
        properties: { customer_id: { type: "string" } },
        required: ["customer_id"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "get_segment_benchmark",
      description: "Returns average metrics for a customer segment to compare against the individual.",
      parameters: {
        type: "object",
        properties: { segment: { type: "string" } },
        required: ["segment"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "calculate_intervention_roi",
      description: "Calculates net ROI of an intervention given uplift score, CLV, and cost.",
      parameters: {
        type: "object",
        properties: {
          uplift_score: { type: "number" },
          clv: { type: "number" },
          intervention_cost: { type: "number" },
        },
        required: ["uplift_score", "clv", "intervention_cost"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "lookup_customer_details",
      description: "Looks up all available details for a customer by ID.",
      parameters: {
        type: "object",
        properties: { customer_id: { type: "string" } },
        required: ["customer_id"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "search_retention_playbook",
      description: "Searches the retention playbook for the best intervention for a given risk factor.",
      parameters: {
        type: "object",
        properties: { risk_factor: { type: "string" } },
        required: ["risk_factor"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "get_all_segment_benchmarks",
      description: "Returns churn metrics and risk profiles for ALL segments at once. Use this when the user asks about 'all segments', wants to compare segments, or asks about overall/cross-segment trends — do NOT call get_segment_benchmark with segment='all'.",
      parameters: { type: "object", properties: {} },
    },
  },
  {
    type: "function",
    function: {
      name: "get_past_interventions",
      description: "Returns all previously generated retention actions and their outcomes for a specific customer. ALWAYS call this before recommending an intervention to avoid repeating failed approaches.",
      parameters: {
        type: "object",
        properties: { customer_id: { type: "string", description: "The customer ID to look up history for." } },
        required: ["customer_id"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "get_intervention_success_rates",
      description: "Returns historical retention rates by intervention type across all customers who have outcome data. Use this to recommend the most effective intervention types.",
      parameters: { type: "object", properties: {} },
    },
  },
  {
    type: "function",
    function: {
      name: "get_at_risk_customers",
      description: "Returns the top high-risk customers by churn probability. Optionally filter by a specific segment. Use when the manager asks who to prioritise or who to contact.",
      parameters: {
        type: "object",
        properties: {
          segment: { type: "string", description: "Optional segment name to filter by. Omit for all segments." },
          limit: { type: "number", description: "Max customers to return (default 10, max 25)." },
        },
      },
    },
  },
  {
    type: "function",
    function: {
      name: "get_revenue_at_risk",
      description: "Estimates total revenue at risk from predicted churners using expected churn probability × assumed CLV. Optionally filter by segment.",
      parameters: {
        type: "object",
        properties: {
          segment: { type: "string", description: "Optional segment name. Omit for all segments." },
        },
      },
    },
  },
  {
    type: "function",
    function: {
      name: "save_retention_action",
      description: "Saves a retention action plan to the database. Only call this when the user explicitly asks to save or schedule the intervention. Returns the saved action ID.",
      parameters: {
        type: "object",
        properties: {
          customer_id:        { type: "string" },
          intervention_type:  { type: "string" },
          channel:            { type: "string" },
          timing:             { type: "string" },
          message_framing:    { type: "string" },
          confidence:         { type: "string" },
        },
        required: ["customer_id", "intervention_type", "channel", "timing", "message_framing"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "get_unactioned_persuadables",
      description: "Returns persuadable customers who have NOT yet had a retention action generated — the highest-priority untouched leads ordered by net ROI.",
      parameters: {
        type: "object",
        properties: {
          limit: { type: "number", description: "Max customers to return (default 10)." },
        },
      },
    },
  },
];

// ─── Tool execution ────────────────────────────────────────────────────────────

/**
 * Turn a Supabase error into something the model is told is a *failure*, not an
 * absence.
 *
 * Nearly every tool below used to destructure only `{ data }`, so an
 * unreachable database produced `data === null` and the tool answered
 * "Customer 1234 not found". The agent then told a CSM the customer does not
 * exist and moved on — a confident wrong answer generated from an outage. The
 * dashboard pages already learned this lesson during the Supabase pause; the
 * agent had not.
 */
function dbFailure(context: string, error: { message: string } | null) {
  console.error(`agent tool: ${context} —`, error?.message);
  return {
    error: `The customer database could not be reached while running ${context}. ` +
      `This is a data-source failure, not a statement that the record is missing — ` +
      `say so rather than concluding anything about the customer. (${error?.message})`,
  };
}

async function executeTool(name: string, args: Record<string, unknown>): Promise<unknown> {
  if (name === "get_top_churn_drivers") {
    const { data, error } = await supabase
      .from("customers")
      .select("customer_id, top_shap_features, churn_probability, segment")
      .eq("customer_id", String(args.customer_id))
      .maybeSingle();
    if (error) return dbFailure("get_top_churn_drivers", error);
    if (!data) return { error: `Customer ${args.customer_id} not found` };
    const shap = (data.top_shap_features as Record<string, number>) ?? {};
    const sorted = Object.entries(shap).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1])).slice(0, 6);
    return {
      customer_id: data.customer_id,
      primary_driver: sorted[0]?.[0] ?? "Unknown",
      top_risk_factors: sorted.map(([feature, shap_value]) => ({
        feature,
        shap_value: Math.round(shap_value * 10000) / 10000,
        effect: shap_value > 0 ? "increases churn risk" : "decreases churn risk",
        magnitude: Math.abs(shap_value) > 0.1 ? "strong" : Math.abs(shap_value) > 0.05 ? "moderate" : "weak",
      })),
    };
  }

  if (name === "get_segment_benchmark") {
    // Aggregated in Postgres, not here.
    //
    // This used to SELECT every row of the segment — up to ~13,000 — into the
    // serverless function and average them in JavaScript, reporting
    // `data.length` as the segment's customer count. That makes the number the
    // agent quotes a property of however many rows PostgREST felt like
    // returning: raise `max-rows` and it is right, lower it and the agent
    // confidently reports a segment of exactly 1,000 customers. The RPC beside
    // it, get_all_segment_benchmarks, already computed the true figures, so the
    // same agent could contradict itself in one conversation.
    const { data, error } = await supabase.rpc("get_segment_summary");
    if (error) return { error: `Could not fetch segment data: ${error.message}` };

    const row = ((data ?? []) as Record<string, unknown>[]).find(
      (s) => String(s.segment) === String(args.segment)
    );
    if (!row) {
      const known = ((data ?? []) as Record<string, unknown>[]).map((s) => s.segment);
      return { error: `Segment ${args.segment} not found. Known segments: ${known.join(", ")}` };
    }

    const r3 = (v: unknown) => Math.round(Number(v) * 1000) / 1000;
    return {
      segment: row.segment,
      customer_count: row.customer_count,
      churn_rate: r3(row.churn_rate),
      avg_churn_probability: r3(row.avg_churn_prob),
      pct_high_risk: r3(row.high_risk_pct),
      pct_persuadable: r3(row.persuadable_pct),
      avg_tenure: r3(row.avg_tenure),
      avg_satisfaction: r3(row.avg_satisfaction),
    };
  }

  if (name === "calculate_intervention_roi") {
    const uplift = Number(args.uplift_score);
    const clv = Number(args.clv);
    const cost = Number(args.intervention_cost);
    const expected = uplift * clv;
    const net = expected - cost;
    return {
      expected_value: Math.round(expected * 100) / 100,
      net_roi: Math.round(net * 100) / 100,
      roi_pct: cost > 0 ? Math.round((net / cost) * 10000) / 100 : 0,
      recommendation: net > 0 ? "Proceed — intervention is financially justified." : "Do not proceed — expected value is below cost.",
    };
  }

  if (name === "lookup_customer_details") {
    const { data, error } = await supabase
      .from("customers")
      .select("*")
      .eq("customer_id", String(args.customer_id))
      .maybeSingle();
    if (error) return dbFailure("lookup_customer_details", error);
    if (!data) return { error: `Customer ${args.customer_id} not found` };
    return data;
  }

  if (name === "search_retention_playbook") {
    const riskFactor = String(args.risk_factor).toLowerCase();
    const { data: rows } = await supabase
      .from("retention_playbook")
      .select("risk_factor_keyword, intervention, message, cost");
    const entries = (rows ?? []) as { risk_factor_keyword: string; intervention: string; message: string; cost: string }[];
    // Longest keyword first, so "payment failure" beats a row keyed on "pay".
    // `find` returns whichever row the database happened to hand back first,
    // which made the chosen intervention depend on physical row order.
    const match = entries
      .filter((r) => r.risk_factor_keyword !== "default" && riskFactor.includes(r.risk_factor_keyword))
      .sort((a, b) => b.risk_factor_keyword.length - a.risk_factor_keyword.length)[0];
    const fallback = entries.find((r) => r.risk_factor_keyword === "default");
    const result = match ?? fallback;
    return result
      ? { intervention: result.intervention, message: result.message, cost: result.cost }
      : { intervention: "Personalized Content", message: "Send a personalised re-engagement email with relevant content.", cost: "Low ($1–5)" };
  }

  if (name === "get_all_segment_benchmarks") {
    const { data, error } = await supabase.rpc("get_segment_summary");
    if (error || !data) return { error: "Could not fetch segment data" };
    return (data as Record<string, unknown>[]).map((s) => ({
      segment: s.segment,
      customer_count: s.customer_count,
      churn_rate: `${(Number(s.churn_rate) * 100).toFixed(1)}%`,
      avg_churn_prob: `${(Number(s.avg_churn_prob) * 100).toFixed(1)}%`,
      high_risk_pct: `${(Number(s.high_risk_pct) * 100).toFixed(1)}%`,
      persuadable_pct: `${(Number(s.persuadable_pct) * 100).toFixed(1)}%`,
    }));
  }

  if (name === "get_past_interventions") {
    const { data: actions, error: actionsError } = await supabase
      .from("retention_actions")
      .select("id, intervention_type, channel, timing, generated_at")
      .eq("customer_id", String(args.customer_id))
      .order("generated_at", { ascending: false })
      .limit(10);
    if (actionsError) return dbFailure("get_past_interventions", actionsError);
    if (!actions || actions.length === 0)
      return { message: `No previous interventions for customer ${args.customer_id}. This would be the first.` };
    const actionIds = (actions as Record<string, unknown>[]).map((a) => a.id as string);
    const { data: feedback } = await supabase
      .from("intervention_feedback")
      .select("retention_action_id, outcome")
      .in("retention_action_id", actionIds);
    const fbMap: Record<string, string> = Object.fromEntries(
      ((feedback ?? []) as Record<string, unknown>[]).map((f) => [f.retention_action_id as string, f.outcome as string])
    );
    return (actions as Record<string, unknown>[]).map((a) => ({
      intervention_type: a.intervention_type,
      channel: a.channel,
      timing: a.timing,
      generated_at: a.generated_at,
      outcome: fbMap[a.id as string] ?? "pending",
    }));
  }

  if (name === "get_intervention_success_rates") {
    const [{ data: actions }, { data: feedback }] = await Promise.all([
      supabase.from("retention_actions").select("id, intervention_type"),
      supabase.from("intervention_feedback").select("retention_action_id, outcome"),
    ]);
    if (!actions) return { error: "No action data available" };
    const fbMap: Record<string, string> = Object.fromEntries(
      ((feedback ?? []) as Record<string, unknown>[]).map((f) => [f.retention_action_id as string, f.outcome as string])
    );
    const byType: Record<string, { total: number; retained: number; withFeedback: number }> = {};
    for (const a of actions as Record<string, unknown>[]) {
      const t = String(a.intervention_type ?? "Unknown");
      if (!byType[t]) byType[t] = { total: 0, retained: 0, withFeedback: 0 };
      byType[t].total++;
      const outcome = fbMap[a.id as string];
      if (outcome) { byType[t].withFeedback++; if (outcome === "retained") byType[t].retained++; }
    }
    return Object.entries(byType)
      .map(([type, v]) => ({
        intervention_type: type,
        total_actions: v.total,
        with_feedback: v.withFeedback,
        retention_rate: v.withFeedback > 0 ? `${Math.round((v.retained / v.withFeedback) * 100)}%` : "No outcome data yet",
      }))
      .sort((a, b) => b.total_actions - a.total_actions);
  }

  if (name === "get_at_risk_customers") {
    const limit = Math.min(Number(args.limit ?? 10), 25);
    let query = supabase
      .from("customers")
      .select("customer_id, segment, churn_probability, risk_tier, customer_type, uplift_score, net_roi")
      .eq("risk_tier", "High Risk")
      .order("churn_probability", { ascending: false })
      .limit(limit);
    if (args.segment) query = query.eq("segment", String(args.segment));
    const { data, error } = await query;
    if (error) return dbFailure("get_at_risk_customers", error);
    if (!data || data.length === 0) return { message: "No high-risk customers found for the given filter." };
    return (data as Record<string, unknown>[]).map((c) => ({
      customer_id: c.customer_id,
      segment: c.segment,
      churn_probability: `${(Number(c.churn_probability) * 100).toFixed(1)}%`,
      customer_type: c.customer_type,
      uplift_score: `${(Number(c.uplift_score) * 100).toFixed(2)}%`,
      net_roi: `$${Number(c.net_roi).toFixed(2)}`,
    }));
  }

  if (name === "get_revenue_at_risk") {
    const { data: clvRow } = await supabase
      .from("business_config")
      .select("value")
      .eq("key", "assumed_clv_usd")
      .single();
    const ASSUMED_CLV = Number((clvRow as { value: string } | null)?.value ?? 500);
    if (args.segment) {
      const { data } = await supabase.rpc("get_churn_kpis", { p_segment: String(args.segment) });
      if (!data) return { error: `No data for segment: ${args.segment}` };
      const k = (Array.isArray(data) ? data[0] : data) as { total: number; avg_churn_prob: number };
      const expected = Math.round(k.total * k.avg_churn_prob);
      return {
        segment: args.segment,
        total_customers: k.total,
        avg_churn_prob: `${(k.avg_churn_prob * 100).toFixed(1)}%`,
        expected_churners: expected,
        estimated_revenue_at_risk: `$${(expected * ASSUMED_CLV).toLocaleString()}`,
        note: `Assumes $${ASSUMED_CLV} CLV per customer.`,
      };
    } else {
      const { data } = await supabase.rpc("get_segment_summary");
      if (!data) return { error: "Could not fetch segment summary" };
      let totalExpected = 0;
      const breakdown = (data as Record<string, unknown>[]).map((s) => {
        const exp = Math.round(Number(s.customer_count) * Number(s.avg_churn_prob));
        totalExpected += exp;
        return { segment: s.segment, expected_churners: exp, revenue_at_risk: `$${(exp * ASSUMED_CLV).toLocaleString()}` };
      });
      return {
        total_expected_churners: totalExpected,
        total_revenue_at_risk: `$${(totalExpected * ASSUMED_CLV).toLocaleString()}`,
        by_segment: breakdown,
        note: `Assumes $${ASSUMED_CLV} CLV per customer.`,
      };
    }
  }

  if (name === "save_retention_action") {
    const customer_id = String(args.customer_id);
    const { data: customer, error: customerError } = await supabase
      .from("customers")
      .select("segment, churn_probability, uplift_score, net_roi")
      .eq("customer_id", customer_id)
      .maybeSingle();
    // Refuse rather than writing an audit row with null segment / churn / ROI,
    // which afterwards is indistinguishable from a plan made without them.
    if (customerError) return dbFailure("save_retention_action", customerError);
    if (!customer) {
      return { error: `Customer ${customer_id} does not exist, so no retention action was saved.` };
    }
    const { data, error } = await supabaseAdmin
      .from("retention_actions")
      .insert({
        id: crypto.randomUUID(),
        customer_id,
        segment: (customer as Record<string, unknown> | null)?.segment ?? null,
        churn_probability: (customer as Record<string, unknown> | null)?.churn_probability ?? null,
        uplift_score: (customer as Record<string, unknown> | null)?.uplift_score ?? null,
        net_roi: (customer as Record<string, unknown> | null)?.net_roi ?? null,
        intervention_type: String(args.intervention_type),
        channel: String(args.channel),
        timing: String(args.timing),
        message_framing: String(args.message_framing),
        confidence: args.confidence != null ? String(args.confidence) : null,
        agent_reasoning: null,
        agentic_mode: true,
      })
      .select("id")
      .single();
    if (error) return { error: `Failed to save action: ${error.message}` };
    return {
      success: true,
      action_id: (data as Record<string, unknown>)?.id,
      message: `Retention action saved for customer ${customer_id}. It will appear in Audit & Analytics.`,
    };
  }

  if (name === "get_unactioned_persuadables") {
    const limit = Math.min(Number(args.limit ?? 10), 25);
    const { data: persuadables, error: persuadablesError } = await supabase
      .from("customers")
      .select("customer_id, segment, churn_probability, uplift_score, net_roi")
      .eq("customer_type", "Persuadable")
      .order("net_roi", { ascending: false })
      .limit(100);
    if (persuadablesError) return dbFailure("get_unactioned_persuadables", persuadablesError);
    if (!persuadables || persuadables.length === 0) return { message: "No persuadable customers found." };
    const ids = (persuadables as Record<string, unknown>[]).map((p) => p.customer_id as string);
    const { data: actioned } = await supabase
      .from("retention_actions")
      .select("customer_id")
      .in("customer_id", ids);
    const actionedSet = new Set(((actioned ?? []) as Record<string, unknown>[]).map((a) => a.customer_id as string));
    const unactioned = (persuadables as Record<string, unknown>[])
      .filter((p) => !actionedSet.has(p.customer_id as string))
      .slice(0, limit);
    if (unactioned.length === 0) return { message: "All top persuadables already have actions generated." };
    return unactioned.map((c) => ({
      customer_id: c.customer_id,
      segment: c.segment,
      churn_probability: `${(Number(c.churn_probability) * 100).toFixed(1)}%`,
      uplift_score: `${(Number(c.uplift_score) * 100).toFixed(2)}%`,
      net_roi: `$${Number(c.net_roi).toFixed(2)}`,
    }));
  }

  return { error: `Unknown tool: ${name}` };
}

// ─── System prompt ─────────────────────────────────────────────────────────────

const TOOL_RULES = `
TOOL CALLING RULES — follow these exactly:
- ALWAYS call tools using the official API tool_call mechanism. NEVER write <function=...> or any XML/text-based format.
- get_top_churn_drivers, lookup_customer_details, get_past_interventions require a real customer_id. Never call them without one.
- get_segment_benchmark requires a real segment name (e.g. "Loyal Customers"). NEVER pass "all" — use get_all_segment_benchmarks instead.
- get_all_segment_benchmarks, get_intervention_success_rates take NO arguments.
- get_at_risk_customers and get_revenue_at_risk work without a segment (returns all segments); segment is optional.
- save_retention_action: only call this when the user explicitly asks to save or schedule the plan. It writes to the database.
- get_unactioned_persuadables: use when the manager asks who hasn't been contacted yet or needs a priority list.
- get_past_interventions: ALWAYS call this before recommending an intervention for a specific customer.
- If a question needs a customer_id you don't have, ask the user for it rather than inventing one.
`;

// ─── Config fetcher ────────────────────────────────────────────────────────────

interface AgentConfig {
  interventionTypes: string;
  channels: string;
  timingOptions: string;
  assumedClv: number;
}

const CONFIG_FALLBACKS: AgentConfig = {
  interventionTypes: "Discount Offer,Loyalty Reward,Personalized Content,Proactive Support Call,Reactivation Campaign,Premium Upgrade,Service Recovery",
  channels: "In-App Notification,Email,Push Notification,Direct Call,SMS",
  timingOptions: "Immediate,Within 24 hours,Within 1 week,During next session",
  assumedClv: 500,
};

async function fetchAgentConfig(): Promise<AgentConfig> {
  const { data } = await supabase
    .from("business_config")
    .select("key, value")
    .in("key", ["intervention_types", "channels", "timing_options", "assumed_clv_usd"]);
  const cfg = Object.fromEntries(
    ((data ?? []) as { key: string; value: string }[]).map((r) => [r.key, r.value])
  );
  return {
    interventionTypes: cfg.intervention_types ?? CONFIG_FALLBACKS.interventionTypes,
    channels: cfg.channels ?? CONFIG_FALLBACKS.channels,
    timingOptions: cfg.timing_options ?? CONFIG_FALLBACKS.timingOptions,
    assumedClv: Number(cfg.assumed_clv_usd ?? CONFIG_FALLBACKS.assumedClv),
  };
}

function fmtList(csv: string, sep = " / ") {
  return csv.split(",").map((s) => s.trim()).join(sep);
}

function buildSystemBatch(cfg: AgentConfig): string {
  return `You are a Customer Success AI generating structured retention action plans.

Use the available tools to gather all necessary data before making a recommendation.
${TOOL_RULES}
After calling all relevant tools, output your final recommendation as valid JSON ONLY (no other text):

{
  "primary_risk_reason": "One sentence describing the main churn driver from SHAP analysis",
  "customer_receptivity": "Assessment of whether intervention will work and why",
  "intervention_type": "One of: ${fmtList(cfg.interventionTypes)}",
  "channel": "One of: ${fmtList(cfg.channels)}",
  "timing": "One of: ${fmtList(cfg.timingOptions)}",
  "message_framing": "2-3 sentence customer-facing message (do not mention ML, churn, or scores)",
  "intervention_cost_estimate": "Low ($1-5) / Medium ($10-25) / High ($50-100)",
  "expected_outcome": "Brief prediction of result if intervention is executed",
  "confidence": "High / Medium / Low"
}

If the customer type is Lost Cause or Sleeping Dog, output:
{"do_not_intervene_reason": "explanation of why intervention would be counterproductive"}`;
}

function buildSystemChat(cfg: AgentConfig): string {
  return `You are an AI Customer Success Assistant. You help Customer Success Managers understand customers and decide what retention actions to take.
${TOOL_RULES}
Valid intervention types: ${fmtList(cfg.interventionTypes, ", ")}.
Be concise and actionable. Use tools to fetch real data before answering. If a question is about all segments or overall trends, use get_all_segment_benchmarks. If a question is about a specific customer, use their customer_id with lookup_customer_details or get_top_churn_drivers.`;
}

// ─── ReAct loop ────────────────────────────────────────────────────────────────

/**
 * Strip reasoning scaffolding some models emit around their answer.
 *
 * `<think>` blocks are a Qwen/DeepSeek-R1 habit and Llama 3.3 does not produce
 * them, but this stays as defence for a model swap. `<function=...>` is the
 * text-shaped tool call the TOOL_CALLING RULES forbid — see runAgentLoop for
 * why removing it needs care rather than just deleting the text.
 */
function stripThinking(text: string): string {
  return text
    .replace(/<think>[\s\S]*?<\/think>/gi, "")
    .replace(/<function=[\s\S]*?<\/function>/gi, "")
    .trim();
}

async function runAgentLoop(
  groq: GroqClient,
  messages: ChatCompletionMessageParam[],
  mode: "batch" | "chat"
): Promise<{ response: string; trace: unknown[]; truncated: boolean }> {
  const trace: unknown[] = [];
  const apiMessages: ChatCompletionMessageParam[] = [...messages];
  let truncated = false;
  const deadline = Date.now() + LOOP_BUDGET_MS;

  for (let round = 1; round <= MAX_ROUNDS; round++) {
    // Stop starting rounds we cannot afford to finish. Running out of rounds
    // lands on the forced summary below, which is a real answer; running out of
    // wall clock lands on a Vercel timeout, which is a blank page.
    if (Date.now() > deadline - REQUEST_TIMEOUT_MS) break;

    const completion = await groq.chat.completions.create({
      model: MODEL,
      messages: apiMessages,
      tools: TOOLS,
      tool_choice: "auto",
      max_tokens: ANSWER_MAX_TOKENS,
    });

    const msg = completion.choices[0].message;
    const finish = completion.choices[0].finish_reason;
    const cleanContent = stripThinking(msg.content ?? "");

    // An answer cut off at the token ceiling is not an answer. In batch mode it
    // is also unparseable JSON, which surfaces to the user as the generic
    // "could not parse agent response" — a message that blames the model for
    // what is really a budget the caller set. Recorded and reported instead.
    if (finish === "length") truncated = true;

    apiMessages.push({
      role: "assistant",
      content: cleanContent,
      tool_calls: msg.tool_calls ?? undefined,
    } as ChatCompletionMessageParam);

    // Finish only when there is nothing left to run AND something was said.
    //
    // Two separate traps here, both of which shipped.
    //
    // The first: this used to also return on `finish_reason === "stop"`, as an
    // OR. When a model returns "stop" *alongside* tool calls — which happens —
    // the loop returned immediately, threw the tool calls away, and handed back
    // whatever text was in that message. That text is usually empty when a model
    // is calling tools, so the user got a blank reply over a trace showing no
    // work done.
    //
    // The second: a message with no tool calls and no content is not an answer,
    // but it satisfies "nothing left to run" and was returned as one. In batch
    // mode that empty string then reached JSON.parse and surfaced as "Could not
    // parse agent response" — a recorded run failed exactly this way, three tool
    // calls deep with an empty reply. Falling through to the forced final call
    // gives the model one more chance under a stricter contract, which is what
    // that call is for.
    if (!msg.tool_calls || msg.tool_calls.length === 0) {
      if (cleanContent.trim()) return { response: cleanContent, trace, truncated };
      break;
    }

    for (const tc of msg.tool_calls) {
      let args: Record<string, unknown> = {};
      try { args = JSON.parse(tc.function.arguments); } catch { /* empty args */ }

      const result = await executeTool(tc.function.name, args);
      trace.push({ round, tool: tc.function.name, args, result });

      apiMessages.push({
        role: "tool",
        tool_call_id: tc.id,
        content: JSON.stringify(result),
      });
    }
  }

  // Out of rounds or out of time — ask for a final answer.
  //
  // Reaching here is normal, not exceptional: a batch plan routinely spends
  // every round on tool calls, so this is the step that actually produces the
  // answer. Getting it right took three attempts, and each failure is worth
  // recording because they all looked like the model misbehaving.
  //
  //  1. Asking for prose in both modes. Batch's contract is a JSON object, so
  //     the model tried to smuggle it out as a tool call named `json`, which
  //     Groq rejects. Two of three recorded runs came back unparseable.
  //  2. Asking for JSON but leaving the tool history in place. With a dozen
  //     tool calls behind it the model simply wanted another one, and omitting
  //     the `tools` array makes Groq infer `tool_choice: "none"` — so a model
  //     that calls a tool anyway fails the entire request with
  //     `400 tool_use_failed`, at the last step, after every tool call has
  //     already been made and paid for.
  //
  // What works is removing the momentum rather than fighting it. The batch
  // final call gets a fresh conversation — the instructions, the original
  // request, and everything the tools returned flattened into one plain-text
  // block — so there is no assistant/tool exchange to continue and nothing to
  // call. `response_format: json_object` then makes the shape the provider's
  // problem instead of a request in a prompt.
  //
  // Chat keeps its tools and `tool_choice: "auto"`, because prose has no shape
  // to enforce and an extra tool call there is harmless.
  const isBatch = mode === "batch";
  let response = "";

  const finalMessages: ChatCompletionMessageParam[] = isBatch
    ? [
        messages[0], // the system prompt, with the JSON schema in it
        ...messages.slice(1).filter((m) => m.role === "user"),
        {
          role: "user",
          content:
            "Here is everything your tools returned:\n\n" +
            JSON.stringify(trace, null, 1).slice(0, 12_000) +
            "\n\nOutput the final recommendation now as the JSON object described " +
            "above. JSON only.",
        },
      ]
    : [
        ...apiMessages,
        {
          role: "user",
          content:
            "Summarise your findings and give a final recommendation now, in " +
            "prose. Do not call any more tools — you have everything you need.",
        },
      ];

  try {
    const final = await groq.chat.completions.create({
      model: MODEL,
      messages: finalMessages,
      ...(isBatch
        ? { response_format: { type: "json_object" as const } }
        : { tools: TOOLS, tool_choice: "auto" as const }),
      max_tokens: ANSWER_MAX_TOKENS,
    });
    if (final.choices[0].finish_reason === "length") truncated = true;
    response = stripThinking(final.choices[0].message.content ?? "");
  } catch (err) {
    console.error("Agent: final summary call failed —", scrub(String(err)));
  }

  // A model that spent its last turn calling tools leaves no prose behind. The
  // work is not lost — the trace is real and the UI renders it — so say what
  // happened rather than returning an empty bubble.
  //
  // Batch mode gets nothing here on purpose: any filler would be parsed as the
  // plan and fail, and the caller's explicit message about an empty response is
  // better than "could not parse".
  if (!response.trim() && !isBatch) {
    const used = [...new Set(trace.map((t) => (t as { tool: string }).tool))];
    response =
      `I gathered data across ${trace.length} tool call${trace.length === 1 ? "" : "s"}` +
      (used.length ? ` (${used.join(", ")})` : "") +
      " but ran out of reasoning rounds before writing a conclusion. The findings " +
      "are in the reasoning trace below — ask a narrower question for a direct answer.";
  }

  return { response, trace, truncated };
}

// ─── Abuse limits ──────────────────────────────────────────────────────────────

/**
 * This route is public and unauthenticated, on purpose — the demo is meant to
 * be usable by anyone who opens the dashboard. But every request spends Groq
 * free-tier tokens (100k/day for the whole project) and can write rows to
 * `retention_actions`, so "anyone" also means "anyone with a for-loop": one
 * script can exhaust the day's quota in a couple of minutes and leave the
 * demo dead for everybody else, with a polluted audit trail behind it.
 *
 * A per-IP bucket held in module scope is a real mitigation and an imperfect
 * one, and it is worth being precise about which. Serverless instances are not
 * shared, so an attacker spread across enough cold starts gets more than the
 * quota below suggests. It stops casual hammering and the accidental infinite
 * retry loop, which is most of what actually happens; it is not a defence
 * against someone who wants the quota gone. The real fix is a shared store
 * (Vercel KV, Upstash) and that is a dependency this project does not need yet.
 */
const RATE_LIMIT = { windowMs: 10 * 60_000, maxRequests: 60 };
const hits = new Map<string, number[]>();

function rateLimited(ip: string): boolean {
  // Generous, because visitors spend their own quota — but not absent. The
  // limit is no longer about protecting a token budget; it stops this route
  // being used as a general-purpose Groq proxy, and stops a script filling
  // `retention_actions` with junk.
  const max = RATE_LIMIT.maxRequests;
  const now = Date.now();
  const cutoff = now - RATE_LIMIT.windowMs;
  const recent = (hits.get(ip) ?? []).filter((t) => t > cutoff);
  recent.push(now);
  hits.set(ip, recent);

  // Keep the map from growing without bound across a long-lived instance.
  if (hits.size > 5000) {
    for (const [key, times] of hits) {
      if (times.every((t) => t <= cutoff)) hits.delete(key);
    }
  }
  return recent.length > max;
}

/** Longest conversation the client may replay back at us. */
const MAX_HISTORY_MESSAGES = 12;

/**
 * The client sends `history` and it is passed straight to the model, so without
 * this a caller can inject their own `system` message and rewrite the agent's
 * instructions — including the rule that only lets it save when asked. Only the
 * two roles a transcript legitimately contains are kept, and only the tail of
 * it, so a long session cannot be replayed back as an unbounded token bill.
 */
function sanitiseHistory(history: ChatCompletionMessageParam[] | undefined) {
  return (history ?? [])
    .filter((m) => m.role === "user" || m.role === "assistant")
    .slice(-MAX_HISTORY_MESSAGES);
}

// ─── Route handler ─────────────────────────────────────────────────────────────

export async function POST(req: NextRequest) {
  try {
    const userKey = readUserKey(req);
    if (!userKey) {
      const supplied = req.headers.get("x-groq-key")?.trim();
      return NextResponse.json(
        {
          error: supplied
            ? "That does not look like a Groq key — they start with `gsk_`. Copy it " +
              "whole from console.groq.com."
            : "The agent runs on your own Groq key. Add one to use it; everything else " +
              "on this dashboard works without it.",
          needs_key: true,
        },
        { status: 401 }
      );
    }

    // One client, for this request only. Never cached, never reused across
    // visitors, gone when the handler returns.
    const groq = new GroqClient({
      apiKey: userKey,
      timeout: REQUEST_TIMEOUT_MS,
      maxRetries: REQUEST_RETRIES,
    });

    // Vercel sets x-forwarded-for; the first entry is the client.
    const ip = (req.headers.get("x-forwarded-for") ?? "unknown").split(",")[0].trim();
    if (rateLimited(ip)) {
      return NextResponse.json(
        {
          error: `Rate limit reached — ${RATE_LIMIT.maxRequests} agent requests per ` +
            `${RATE_LIMIT.windowMs / 60_000} minutes.`,
        },
        { status: 429 }
      );
    }

    const body = await req.json();
    const { mode, customer, message, history } = body as {
      mode: "batch" | "chat";
      customer?: Record<string, unknown>;
      message?: string;
      history?: ChatCompletionMessageParam[];
    };

    const cfg = await fetchAgentConfig();
    let messages: ChatCompletionMessageParam[];

    if (mode === "batch" && customer) {
      messages = [
        { role: "system", content: buildSystemBatch(cfg) },
        {
          role: "user",
          content: `Generate a retention action plan for customer ${customer.customer_id}. ` +
            `Segment: '${customer.segment}', churn probability: ${(Number(customer.churn_probability) * 100).toFixed(1)}%, ` +
            `uplift score: ${Number(customer.uplift_score).toFixed(3)}, ` +
            `customer type: '${customer.customer_type}'. Assumed CLV: $${cfg.assumedClv}. Use tools to gather supporting data first.`,
        },
      ];
    } else if (mode === "chat" && message) {
      messages = [
        { role: "system", content: buildSystemChat(cfg) },
        ...sanitiseHistory(history),
        { role: "user", content: message },
      ];
    } else {
      return NextResponse.json({ error: "Invalid request" }, { status: 400 });
    }

    const { response, trace, truncated } = await runAgentLoop(groq, messages, mode);

    if (mode === "batch") {
      let raw = response;
      if (raw.includes("```json")) raw = raw.split("```json")[1].split("```")[0].trim();
      else if (raw.includes("```")) raw = raw.split("```")[1].split("```")[0].trim();
      try {
        const action = JSON.parse(raw);
        action.customer_id = customer?.customer_id;
        action.segment = customer?.segment;
        action.churn_probability = customer?.churn_probability;
        action.uplift_score = customer?.uplift_score;
        action.net_roi = customer?.net_roi;
        action.trace = trace;

        // Save to retention_actions server-side (uses service role key to bypass RLS)
        let savedId: string | null = null;
        let saveError: string | null = null;
        if (!action.do_not_intervene_reason && customer) {
          const { data: saved, error: saveErr } = await supabaseAdmin
            .from("retention_actions")
            .insert({
              id: crypto.randomUUID(),
              customer_id: String(customer.customer_id),
              segment: customer.segment ?? null,
              churn_probability: customer.churn_probability ?? null,
              uplift_score: customer.uplift_score ?? null,
              net_roi: customer.net_roi ?? null,
              intervention_type: action.intervention_type ?? null,
              channel: action.channel ?? null,
              timing: action.timing ?? null,
              message_framing: action.message_framing ?? null,
              confidence: action.confidence ?? null,
              agent_reasoning: trace,
              agentic_mode: true,
            })
            .select("id")
            .single();
          if (saveErr) {
            console.error("retention_actions insert failed:", saveErr.message);
            // Travels back to the browser so the UI can say the plan was not
            // recorded. Logging alone is what let this run unnoticed.
            saveError = HAS_SERVICE_ROLE
              ? saveErr.message
              : "SUPABASE_SERVICE_ROLE_KEY is not set on this deployment, so the " +
                "plan could not be written to retention_actions.";
          } else {
            savedId = (saved as { id: string } | null)?.id ?? null;
          }
        }

        return NextResponse.json({ action, saved_id: savedId, save_error: saveError, truncated });
      } catch {
        return NextResponse.json({
          action: {
            error: truncated
              ? `The agent's reply hit the ${ANSWER_MAX_TOKENS}-token ceiling and stopped mid-JSON, so there is no plan to show. This is a budget limit, not a model failure.`
              : "Could not parse agent response",
            raw: raw.slice(0, 400),
            trace,
          },
          truncated,
        });
      }
    }

    return NextResponse.json({ response, trace, truncated });
  } catch (err: unknown) {
    // A spent quota is an ordinary, expected outcome on a free tier and is not
    // the same thing as the agent being broken, so it is reported as itself
    // rather than as a stack trace.
    if (isQuotaError(err)) {
      console.error("Agent: visitor key is rate limited or out of quota");
      return NextResponse.json(
        {
          error: "Groq is rate limiting that key, or its daily quota is spent. The key " +
            "itself is fine — the free tier resets daily.",
          quota_exhausted: true,
        },
        { status: 429 }
      );
    }

    if (isRetiredModel(err)) {
      console.error(`Agent: Groq no longer serves ${MODEL} — update AGENT_MODEL in src/lib/models.ts`);
      return NextResponse.json(
        {
          error: `This deployment is configured to use \`${MODEL}\`, which Groq no ` +
            "longer serves. That is a configuration problem here, not a problem with " +
            "your key — hosted model names get retired, and this one has been.",
          retired_model: true,
        },
        { status: 503 }
      );
    }

    if (isAuthError(err)) {
      console.error("Agent: Groq rejected the visitor's key");
      return NextResponse.json(
        {
          error: "Groq rejected that key. Check it was copied whole from console.groq.com.",
          invalid_key: true,
          needs_key: true,
        },
        { status: 401 }
      );
    }

    // scrub, because this path returns the message to the browser and a
    // provider SDK can echo request context into it.
    console.error("Agent error:", scrub(String(err)));
    return NextResponse.json({ error: scrub(String(err)) }, { status: 500 });
  }
}
