import { NextRequest, NextResponse } from "next/server";
import GroqClient from "groq-sdk";
import type { ChatCompletionMessageParam, ChatCompletionTool } from "groq-sdk/resources/chat/completions";
import { createClient } from "@supabase/supabase-js";

const groq = new GroqClient({ apiKey: process.env.GROQ_API_KEY || "" });
const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL || "https://placeholder.supabase.co",
  process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY || "placeholder-key"
);

const MODEL = "llama-3.3-70b-versatile";
const MAX_ROUNDS = 5;

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

async function executeTool(name: string, args: Record<string, unknown>): Promise<unknown> {
  if (name === "get_top_churn_drivers") {
    const { data } = await supabase
      .from("customers")
      .select("customer_id, top_shap_features, churn_probability, segment")
      .eq("customer_id", String(args.customer_id))
      .single();
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
    const { data } = await supabase
      .from("customers")
      .select("churn_probability, uplift_score, risk_tier, customer_type, tenure, satisfaction_score")
      .eq("segment", String(args.segment));
    if (!data || data.length === 0) return { error: `Segment ${args.segment} not found` };
    const avg = (key: string) =>
      Math.round((data.reduce((s: number, r: Record<string, unknown>) => s + ((r[key] as number) ?? 0), 0) / data.length) * 1000) / 1000;
    return {
      segment: args.segment,
      customer_count: data.length,
      avg_churn_probability: avg("churn_probability"),
      avg_uplift_score: avg("uplift_score"),
      pct_high_risk: Math.round(data.filter((r) => r.risk_tier === "High Risk").length / data.length * 1000) / 1000,
      pct_persuadable: Math.round(data.filter((r) => r.customer_type === "Persuadable").length / data.length * 1000) / 1000,
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
    const { data } = await supabase
      .from("customers")
      .select("*")
      .eq("customer_id", String(args.customer_id))
      .single();
    if (!data) return { error: `Customer ${args.customer_id} not found` };
    return data;
  }

  if (name === "search_retention_playbook") {
    const riskFactor = String(args.risk_factor).toLowerCase();
    const playbook: Record<string, unknown> = {
      satisfaction: { intervention: "Proactive Support Call", message: "Reach out to understand and resolve the satisfaction issue before they escalate.", cost: "Medium ($10–25)" },
      complain: { intervention: "Service Recovery", message: "Apologise directly, offer a goodwill gesture, and close the loop on the complaint.", cost: "Medium ($10–25)" },
      days_since_last_order: { intervention: "Reactivation Campaign", message: "Send a personalised win-back offer highlighting new products relevant to past purchases.", cost: "Low ($1–5)" },
      tenure: { intervention: "Loyalty Reward", message: "Recognise their loyalty with an exclusive member reward to reinforce the relationship.", cost: "Low ($1–5)" },
      order_count: { intervention: "Personalized Content", message: "Send curated product recommendations based on their purchase history to re-engage browsing.", cost: "Low ($1–5)" },
      cashback: { intervention: "Discount Offer", message: "Offer a targeted cashback or discount on their next order to incentivise return.", cost: "Low ($1–5)" },
      default: { intervention: "Personalized Content", message: "Send a personalised re-engagement email with relevant content.", cost: "Low ($1–5)" },
    };
    const match = Object.entries(playbook).find(([k]) => riskFactor.includes(k));
    return match ? match[1] : playbook.default;
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
    const { data: actions } = await supabase
      .from("retention_actions")
      .select("id, intervention_type, channel, timing, generated_at")
      .eq("customer_id", String(args.customer_id))
      .order("generated_at", { ascending: false })
      .limit(10);
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
    const { data } = await query;
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
    const ASSUMED_CLV = 500;
    if (args.segment) {
      const { data } = await supabase.rpc("get_churn_kpis", { p_segment: String(args.segment) });
      if (!data) return { error: `No data for segment: ${args.segment}` };
      const k = data as { total: number; avg_churn_prob: number };
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
    const { data: customer } = await supabase
      .from("customers")
      .select("segment, churn_probability, uplift_score, net_roi")
      .eq("customer_id", customer_id)
      .single();
    const { data, error } = await supabase
      .from("retention_actions")
      .insert({
        customer_id,
        segment: (customer as Record<string, unknown> | null)?.segment ?? null,
        churn_probability: (customer as Record<string, unknown> | null)?.churn_probability ?? null,
        uplift_score: (customer as Record<string, unknown> | null)?.uplift_score ?? null,
        net_roi: (customer as Record<string, unknown> | null)?.net_roi ?? null,
        intervention_type: String(args.intervention_type),
        channel: String(args.channel),
        timing: String(args.timing),
        message_framing: String(args.message_framing),
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
    const { data: persuadables } = await supabase
      .from("customers")
      .select("customer_id, segment, churn_probability, uplift_score, net_roi")
      .eq("customer_type", "Persuadable")
      .order("net_roi", { ascending: false })
      .limit(100);
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

const SYSTEM_BATCH = `You are a Customer Success AI generating structured retention action plans.

Use the available tools to gather all necessary data before making a recommendation.
${TOOL_RULES}
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
{"do_not_intervene_reason": "explanation of why intervention would be counterproductive"}`;

const SYSTEM_CHAT = `You are an AI Customer Success Assistant. You help Customer Success Managers understand customers and decide what retention actions to take.
${TOOL_RULES}
Be concise and actionable. Use tools to fetch real data before answering. If a question is about all segments or overall trends, use get_all_segment_benchmarks. If a question is about a specific customer, use their customer_id with lookup_customer_details or get_top_churn_drivers.`;

// ─── ReAct loop ────────────────────────────────────────────────────────────────

function stripThinking(text: string): string {
  return text
    .replace(/<think>[\s\S]*?<\/think>/gi, "")
    .replace(/<function=[\s\S]*?<\/function>/gi, "")
    .trim();
}

async function runAgentLoop(
  messages: ChatCompletionMessageParam[],
  mode: "batch" | "chat"
): Promise<{ response: string; trace: unknown[] }> {
  const trace: unknown[] = [];
  const apiMessages: ChatCompletionMessageParam[] = [...messages];

  for (let round = 1; round <= MAX_ROUNDS; round++) {
    const completion = await groq.chat.completions.create({
      model: MODEL,
      messages: apiMessages,
      tools: TOOLS,
      tool_choice: "auto",
      max_tokens: 4096,
    });

    const msg = completion.choices[0].message;
    const finish = completion.choices[0].finish_reason;
    const cleanContent = stripThinking(msg.content ?? "");

    apiMessages.push({
      role: "assistant",
      content: cleanContent,
      tool_calls: msg.tool_calls ?? undefined,
    } as ChatCompletionMessageParam);

    if (!msg.tool_calls || finish === "stop") {
      return { response: cleanContent, trace };
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

  // Max rounds — force final answer
  const final = await groq.chat.completions.create({
    model: MODEL,
    messages: [...apiMessages, { role: "user", content: "Summarise findings and give a final recommendation now." }],
    max_tokens: 1500,
  });
  return { response: stripThinking(final.choices[0].message.content ?? ""), trace };
}

// ─── Route handler ─────────────────────────────────────────────────────────────

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const { mode, customer, message, history } = body as {
      mode: "batch" | "chat";
      customer?: Record<string, unknown>;
      message?: string;
      history?: ChatCompletionMessageParam[];
    };

    let messages: ChatCompletionMessageParam[];

    if (mode === "batch" && customer) {
      messages = [
        { role: "system", content: SYSTEM_BATCH },
        {
          role: "user",
          content: `Generate a retention action plan for customer ${customer.customer_id}. ` +
            `Segment: '${customer.segment}', churn probability: ${(Number(customer.churn_probability) * 100).toFixed(1)}%, ` +
            `uplift score: ${Number(customer.uplift_score).toFixed(3)}, ` +
            `customer type: '${customer.customer_type}'. CLV assumption: $500. Use tools to gather supporting data first.`,
        },
      ];
    } else if (mode === "chat" && message) {
      messages = [
        { role: "system", content: SYSTEM_CHAT },
        ...(history ?? []),
        { role: "user", content: message },
      ];
    } else {
      return NextResponse.json({ error: "Invalid request" }, { status: 400 });
    }

    const { response, trace } = await runAgentLoop(messages, mode);

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
        return NextResponse.json({ action });
      } catch {
        return NextResponse.json({ action: { error: "Could not parse agent response", raw: raw.slice(0, 400), trace } });
      }
    }

    return NextResponse.json({ response, trace });
  } catch (err: unknown) {
    console.error("Agent error:", err);
    return NextResponse.json({ error: String(err) }, { status: 500 });
  }
}
