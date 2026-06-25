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

  return { error: `Unknown tool: ${name}` };
}

// ─── System prompt ─────────────────────────────────────────────────────────────

const SYSTEM_BATCH = `You are a Customer Success AI generating structured retention action plans.

Use the available tools to gather all necessary data before making a recommendation.

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

const SYSTEM_CHAT = `You are an AI Customer Success Assistant. You help Customer Success Managers understand customers and decide what retention actions to take. Always use tools to get real data. Be concise and actionable.`;

// ─── ReAct loop ────────────────────────────────────────────────────────────────

// qwen-qwq-32b is a reasoning model — it wraps internal thinking in <think>…</think>.
// Strip that block before returning content to callers or storing in message history.
function stripThinking(text: string): string {
  return text.replace(/<think>[\s\S]*?<\/think>/gi, "").trim();
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
