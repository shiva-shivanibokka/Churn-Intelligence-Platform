"use client";

import { useMemo, useState } from "react";
import { PersuadableCustomer } from "@/lib/data";
import { PageTitle, SectionHeading } from "@/components/ui/section-heading";
import { MetricCard } from "@/components/ui/metric-card";

interface Props { persuadables: PersuadableCustomer[] }

type Action = {
  customer_id: string;
  segment: string;
  churn_probability: number;
  uplift_score: number;
  net_roi: number;
  intervention_type?: string;
  channel?: string;
  timing?: string;
  message_framing?: string;
  primary_risk_reason?: string;
  customer_receptivity?: string;
  expected_outcome?: string;
  confidence?: string;
  do_not_intervene_reason?: string;
  error?: string;
  trace?: TraceStep[];
};

type TraceStep = {
  round: number;
  tool: string;
  args: Record<string, unknown>;
  result: Record<string, unknown>;
};

type ChatMessage = { role: string; content: string; trace?: TraceStep[] };

const TOOL_META: Record<string, { label: string; color: string }> = {
  get_top_churn_drivers:      { label: "Top Churn Drivers",       color: "#F43F5E" },
  lookup_customer_details:    { label: "Customer Details",        color: "#6366F1" },
  get_segment_benchmark:      { label: "Segment Benchmark",       color: "#A855F7" },
  get_all_segment_benchmarks: { label: "All Segment Benchmarks",  color: "#A855F7" },
  search_retention_playbook:  { label: "Retention Playbook",      color: "#F59E0B" },
  calculate_intervention_roi: { label: "ROI Calculation",         color: "#10B981" },
};

function renderToolResult(tool: string, result: Record<string, unknown>) {
  if (tool === "get_top_churn_drivers") {
    const factors = result.top_risk_factors as { feature: string; shap_value: number; effect: string; magnitude: string }[] ?? [];
    return (
      <div className="space-y-1.5">
        <p className="text-[11px] font-bold text-[#6B7280] uppercase tracking-wide mb-2">
          Primary driver: <span className="text-[#1E1B4B] normal-case font-semibold">{String(result.primary_driver ?? "—")}</span>
        </p>
        {factors.map((f) => (
          <div key={f.feature} className="flex items-center gap-2">
            <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded shrink-0 ${f.effect.includes("increases") ? "bg-[#FEE2E2] text-[#991B1B]" : "bg-[#D1FAE5] text-[#065F46]"}`}>
              {f.effect.includes("increases") ? "+RISK" : "−RISK"}
            </span>
            <span className="text-[12px] font-semibold text-[#1E1B4B] w-52 shrink-0">{f.feature}</span>
            <span className="text-[11px] text-[#6B7280]">SHAP {f.shap_value.toFixed(4)} · {f.magnitude}</span>
          </div>
        ))}
      </div>
    );
  }
  if (tool === "lookup_customer_details") {
    const fields: [string, string][] = [
      ["Segment", String(result.segment ?? "—")],
      ["Churn Prob", result.churn_probability !== undefined ? `${(Number(result.churn_probability) * 100).toFixed(1)}%` : "—"],
      ["Risk Tier", String(result.risk_tier ?? "—")],
      ["Customer Type", String(result.customer_type ?? "—")],
      ["Uplift Score", result.uplift_score !== undefined ? `${(Number(result.uplift_score) * 100).toFixed(2)}%` : "—"],
      ["Net ROI", result.net_roi !== undefined ? `$${Number(result.net_roi).toFixed(2)}` : "—"],
      ["Tenure", result.tenure !== undefined ? `${result.tenure} months` : "—"],
      ["Satisfaction", String(result.satisfaction_score ?? "—")],
    ];
    return (
      <div className="grid grid-cols-2 gap-x-6 gap-y-1">
        {fields.map(([k, v]) => (
          <div key={k} className="flex gap-1 text-[12px]">
            <span className="text-[#7C3AED] font-semibold shrink-0">{k}:</span>
            <span className="text-[#1E1B4B]">{v}</span>
          </div>
        ))}
      </div>
    );
  }
  if (tool === "get_segment_benchmark") {
    return (
      <div className="grid grid-cols-2 gap-x-6 gap-y-1 text-[12px]">
        {[
          ["Segment", String(result.segment ?? "—")],
          ["Customers", String(result.customer_count ?? "—")],
          ["Avg Churn Prob", result.avg_churn_probability !== undefined ? `${(Number(result.avg_churn_probability) * 100).toFixed(1)}%` : "—"],
          ["Avg Uplift", result.avg_uplift_score !== undefined ? `${(Number(result.avg_uplift_score) * 100).toFixed(2)}%` : "—"],
          ["% High Risk", result.pct_high_risk !== undefined ? `${(Number(result.pct_high_risk) * 100).toFixed(1)}%` : "—"],
          ["% Persuadable", result.pct_persuadable !== undefined ? `${(Number(result.pct_persuadable) * 100).toFixed(1)}%` : "—"],
        ].map(([k, v]) => (
          <div key={k} className="flex gap-1">
            <span className="text-[#7C3AED] font-semibold shrink-0">{k}:</span>
            <span className="text-[#1E1B4B]">{v}</span>
          </div>
        ))}
      </div>
    );
  }
  if (tool === "search_retention_playbook") {
    return (
      <div className="space-y-1 text-[12px]">
        <div className="flex gap-1"><span className="text-[#7C3AED] font-semibold">Recommended:</span><span className="text-[#1E1B4B] font-semibold">{String(result.intervention ?? "—")}</span></div>
        <div className="flex gap-1"><span className="text-[#7C3AED] font-semibold">Cost:</span><span className="text-[#1E1B4B]">{String(result.cost ?? "—")}</span></div>
        <p className="text-[#1E1B4B] italic mt-1">"{String(result.message ?? "")}"</p>
      </div>
    );
  }
  if (tool === "get_all_segment_benchmarks") {
    const rows = Array.isArray(result) ? result as Record<string, unknown>[] : [];
    return (
      <div className="overflow-x-auto">
        <table className="text-[11px] w-full border-collapse">
          <thead><tr className="bg-[#F5F3FF]">
            {["Segment","Customers","Churn Rate","Avg Prob","High Risk %","Persuadable %"].map((h) => (
              <th key={h} className="text-left px-2 py-1 text-[#7C3AED] font-bold border-b border-[#EDE9FE]">{h}</th>
            ))}
          </tr></thead>
          <tbody>{rows.map((r) => (
            <tr key={String(r.segment)} className="border-b border-[#F3F0FF]">
              <td className="px-2 py-1 font-semibold">{String(r.segment)}</td>
              <td className="px-2 py-1">{Number(r.customer_count).toLocaleString()}</td>
              <td className="px-2 py-1">{String(r.churn_rate)}</td>
              <td className="px-2 py-1">{String(r.avg_churn_prob)}</td>
              <td className="px-2 py-1">{String(r.high_risk_pct)}</td>
              <td className="px-2 py-1">{String(r.persuadable_pct)}</td>
            </tr>
          ))}</tbody>
        </table>
      </div>
    );
  }
  if (tool === "calculate_intervention_roi") {
    const net = Number(result.net_roi ?? 0);
    return (
      <div className="space-y-1 text-[12px]">
        <div className="flex gap-4">
          <div><span className="text-[#7C3AED] font-semibold">Expected Value: </span><span className="font-bold">${Number(result.expected_value ?? 0).toFixed(2)}</span></div>
          <div><span className="text-[#7C3AED] font-semibold">Net ROI: </span><span className={`font-bold ${net >= 0 ? "text-[#10B981]" : "text-[#EF4444]"}`}>${net.toFixed(2)}</span></div>
          <div><span className="text-[#7C3AED] font-semibold">ROI %: </span><span className="font-bold">{Number(result.roi_pct ?? 0).toFixed(0)}%</span></div>
        </div>
        <p className={`font-semibold text-[12px] ${net >= 0 ? "text-[#10B981]" : "text-[#EF4444]"}`}>{String(result.recommendation ?? "")}</p>
      </div>
    );
  }
  return <pre className="text-[11px] text-[#6B7280] whitespace-pre-wrap">{JSON.stringify(result, null, 2)}</pre>;
}

function AgentTrace({ trace, defaultOpen = false }: { trace: TraceStep[]; defaultOpen?: boolean }) {
  const [open, setOpen] = useState(defaultOpen);
  if (!trace || trace.length === 0) return null;
  return (
    <div className="border border-[#DDD6FE] rounded-xl overflow-hidden mb-3">
      <button
        onClick={() => setOpen((o) => !o)}
        className="w-full flex items-center gap-2 px-4 py-2.5 bg-[#F5F3FF] hover:bg-[#EDE9FE] transition-colors text-left"
      >
        <span className="text-[#7C3AED] text-[11px]">{open ? "▼" : "▶"}</span>
        <span className="text-[11px] font-bold uppercase tracking-wide text-[#7C3AED]">
          Agent Reasoning — {trace.length} tool call{trace.length !== 1 ? "s" : ""}
        </span>
        <span className="ml-auto text-[10px] text-[#A78BFA]">{open ? "collapse" : "expand"}</span>
      </button>
      {open && (
        <div className="px-4 pt-3 pb-4 bg-white">
          <div className="relative pl-6">
            <div className="absolute left-2.5 top-0 bottom-0 w-px bg-[#DDD6FE]" />
            {trace.map((step, si) => {
              const meta = TOOL_META[step.tool] ?? { label: step.tool, color: "#6B7280" };
              return (
                <div key={si} className="relative mb-4 last:mb-0">
                  <span
                    className="absolute -left-6 top-0 w-5 h-5 rounded-full text-white text-[9px] font-bold flex items-center justify-center shrink-0"
                    style={{ background: meta.color }}
                  >
                    {step.round}
                  </span>
                  <div className="bg-[#FAFAFA] border border-[#EDE9FE] rounded-xl px-4 py-3">
                    <p className="text-[12px] font-bold mb-2" style={{ color: meta.color }}>{meta.label}</p>
                    {renderToolResult(step.tool, step.result)}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}

export function RetentionClient({ persuadables }: Props) {
  const [selectedId, setSelectedId] = useState("");
  const [loading, setLoading] = useState(false);
  const [action, setAction] = useState<Action | null>(null);
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [chatInput, setChatInput] = useState("");
  const [chatLoading, setChatLoading] = useState(false);
  const [tab, setTab] = useState<"batch" | "chat">("batch");

  const selected = useMemo(() => persuadables.find((c) => c.customer_id === selectedId), [persuadables, selectedId]);

  async function generateAction() {
    if (!selected) return;
    setLoading(true);
    setAction(null);
    try {
      const res = await fetch("/api/agent", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mode: "batch", customer: selected }),
      });
      const data = await res.json();
      setAction(data.action ?? { error: data.error });
    } catch (e) {
      setAction({ customer_id: selectedId, segment: "", churn_probability: 0, uplift_score: 0, net_roi: 0, error: String(e) });
    }
    setLoading(false);
  }

  async function sendChat() {
    if (!chatInput.trim()) return;
    const userMsg: ChatMessage = { role: "user", content: chatInput };
    const history = [...chatMessages, userMsg];
    setChatMessages(history);
    setChatInput("");
    setChatLoading(true);
    try {
      const res = await fetch("/api/agent", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mode: "chat", message: chatInput, history: chatMessages.map(({ role, content }) => ({ role, content })) }),
      });
      const data = await res.json();
      const aiMsg: ChatMessage = {
        role: "assistant",
        content: data.response ?? data.error ?? "No response.",
        trace: data.trace?.length ? data.trace : undefined,
      };
      setChatMessages([...history, aiMsg]);
    } catch (e) {
      setChatMessages([...history, { role: "assistant", content: `Error: ${e}` }]);
    }
    setChatLoading(false);
  }

  const confidenceColor = (c?: string) =>
    c === "High" ? "#10B981" : c === "Medium" ? "#F59E0B" : "#EF4444";

  return (
    <div>
      <PageTitle>Retention Actions</PageTitle>

      <div className="flex gap-2 mb-6">
        {(["batch", "chat"] as const).map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`px-5 py-2 rounded-xl font-semibold text-[14px] transition-all ${
              tab === t
                ? "text-white shadow-lg"
                : "bg-white border-2 border-[#DDD6FE] text-[#6B7280] hover:border-[#818CF8]"
            }`}
            style={tab === t ? { background: "linear-gradient(135deg, #6366F1, #4338CA)" } : {}}
          >
            {t === "batch" ? "Generate Action Plan" : "Ask AI Assistant"}
          </button>
        ))}
      </div>

      {tab === "batch" && (
        <div className="space-y-6">
          {/* Customer selector */}
          <SectionHeading>Select a Customer</SectionHeading>
          <div className="flex gap-3 flex-wrap items-end">
            <div className="flex-1 min-w-[260px]">
              <label className="block text-[13px] font-semibold text-[#4F46E5] mb-1.5">
                Customer ID — showing top Persuadables by ROI
              </label>
              <select
                value={selectedId}
                onChange={(e) => { setSelectedId(e.target.value); setAction(null); }}
                className="w-full rounded-xl border-2 border-[#818CF8] bg-white px-4 py-3 text-[14px] text-[#1E1B4B] font-medium focus:outline-none focus:border-[#4F46E5] min-h-[48px]"
              >
                <option value="">— Select customer —</option>
                {persuadables.map((c) => (
                  <option key={c.customer_id} value={c.customer_id}>
                    {c.customer_id} | {c.segment} | Churn {(c.churn_probability * 100).toFixed(1)}% | ROI ${c.net_roi.toFixed(0)}
                  </option>
                ))}
              </select>
            </div>
            <button
              onClick={generateAction}
              disabled={!selectedId || loading}
              className="px-6 py-3 rounded-xl font-bold text-[14px] text-white disabled:opacity-50 disabled:cursor-not-allowed transition-all hover:-translate-y-0.5"
              style={{ background: "linear-gradient(135deg, #6366F1, #4338CA)", boxShadow: "0 4px 16px rgba(79,70,229,0.4)" }}
            >
              {loading ? "Generating…" : "Generate Plan"}
            </button>
          </div>

          {/* Customer summary cards */}
          {selected && (
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
              <MetricCard label="Churn Probability" value={`${(selected.churn_probability * 100).toFixed(1)}%`} accentColor="#EF4444" />
              <MetricCard label="Uplift Score" value={`${selected.uplift_score >= 0 ? "+" : ""}${(selected.uplift_score * 100).toFixed(2)}%`} accentColor="#4F46E5" />
              <MetricCard label="Net ROI" value={`$${selected.net_roi.toFixed(2)}`} accentColor="#10B981" />
              <MetricCard label="Customer Type" value={selected.customer_type} accentColor="#7C3AED" />
            </div>
          )}

          {/* Action card */}
          {action && !action.error && !action.do_not_intervene_reason && (
            <div>
              <SectionHeading>AI-Generated Retention Plan</SectionHeading>
              {action.trace && action.trace.length > 0 && (
                <div className="mb-4">
                  <AgentTrace trace={action.trace} defaultOpen />
                </div>
              )}
              <div className="bg-white rounded-2xl border-2 border-[#DDD6FE] p-6 shadow-sm space-y-4">
                <div className="flex items-center justify-between flex-wrap gap-2">
                  <span className="text-[15px] font-bold text-[#1E1B4B]">{action.intervention_type}</span>
                  <span
                    className="px-3 py-1 rounded-full text-[12px] font-bold text-white"
                    style={{ background: confidenceColor(action.confidence) }}
                  >
                    {action.confidence} Confidence
                  </span>
                </div>
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-[14px]">
                  <div><p className="text-[11px] font-bold uppercase tracking-wide text-[#7C3AED]">Channel</p><p className="font-semibold">{action.channel}</p></div>
                  <div><p className="text-[11px] font-bold uppercase tracking-wide text-[#7C3AED]">Timing</p><p className="font-semibold">{action.timing}</p></div>
                  <div><p className="text-[11px] font-bold uppercase tracking-wide text-[#7C3AED]">Cost</p><p className="font-semibold">{(action as Record<string, unknown>).intervention_cost_estimate as string}</p></div>
                </div>
                <div className="bg-[#F5F3FF] rounded-xl p-4 border-l-4 border-[#4F46E5]">
                  <p className="text-[12px] font-bold uppercase tracking-wide text-[#4F46E5] mb-1">Customer Message</p>
                  <p className="text-[14px] text-[#1E1B4B] leading-relaxed">{action.message_framing}</p>
                </div>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-[14px]">
                  <div>
                    <p className="text-[11px] font-bold uppercase tracking-wide text-[#7C3AED] mb-1">Primary Risk</p>
                    <p>{action.primary_risk_reason}</p>
                  </div>
                  <div>
                    <p className="text-[11px] font-bold uppercase tracking-wide text-[#7C3AED] mb-1">Expected Outcome</p>
                    <p>{action.expected_outcome}</p>
                  </div>
                </div>
              </div>
            </div>
          )}

          {action?.do_not_intervene_reason && (
            <div className="bg-[#FEF2F2] border-2 border-[#FECACA] rounded-2xl p-5">
              <p className="font-bold text-[#EF4444] mb-1">Do Not Intervene</p>
              <p className="text-[14px] text-[#7F1D1D]">{action.do_not_intervene_reason}</p>
            </div>
          )}

          {action?.error && (
            <div className="bg-[#FEF2F2] border-2 border-[#FECACA] rounded-2xl p-5">
              <p className="font-bold text-[#EF4444]">Error: {action.error}</p>
            </div>
          )}
        </div>
      )}

      {tab === "chat" && (
        <div className="space-y-4">
          <SectionHeading>Ask Your AI Customer Success Assistant</SectionHeading>
          <p className="text-[14px] text-[#6B7280]">
            Ask anything about your customers: &quot;Tell me about customer 50123&quot;, &quot;Which At-Risk customers are most urgent?&quot;, &quot;Is it worth offering a discount to customer 50456?&quot;
          </p>

          {/* Chat history */}
          <div className="space-y-3 max-h-[500px] overflow-y-auto">
            {chatMessages.length === 0 && (
              <div className="text-center py-12 text-[#A78BFA] text-[14px]">
                No messages yet — ask a question below.
              </div>
            )}
            {chatMessages.map((m, i) => (
              <div
                key={i}
                className={`rounded-2xl px-5 py-4 text-[14px] border-2 ${
                  m.role === "user"
                    ? "bg-[#EEF2FF] border-[#C7D2FE] ml-8"
                    : "bg-white border-[#DDD6FE] mr-8"
                }`}
              >
                <p className="text-[11px] font-bold uppercase tracking-wide text-[#7C3AED] mb-1">
                  {m.role === "user" ? "You" : "AI Assistant"}
                </p>
                {m.role === "assistant" && m.trace && <AgentTrace trace={m.trace} />}
                <p className="text-[#1E1B4B] whitespace-pre-wrap leading-relaxed">{m.content}</p>
              </div>
            ))}
            {chatLoading && (
              <div className="bg-white border-2 border-[#DDD6FE] rounded-2xl px-5 py-4 mr-8">
                <p className="text-[11px] font-bold uppercase tracking-wide text-[#7C3AED] mb-1">AI Assistant</p>
                <p className="text-[#A78BFA] text-[14px]">Thinking…</p>
              </div>
            )}
          </div>

          {/* Suggested prompts when empty */}
          {chatMessages.length === 0 && (
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 mt-2 mb-4">
              {[
                "Tell me about customer 50001",
                "Which At-Risk customers are most urgent?",
                "Is it worth offering a discount to high-churn Lapsed customers?",
                "What are the top churn drivers across all segments?",
              ].map((prompt) => (
                <button
                  key={prompt}
                  onClick={() => { setChatInput(prompt); }}
                  className="text-left px-4 py-3 rounded-xl border-2 border-[#DDD6FE] bg-white text-[13px] text-[#6366F1] font-medium hover:border-[#6366F1] transition-all"
                >
                  {prompt}
                </button>
              ))}
            </div>
          )}

          {/* Input */}
          <div className="flex gap-3 mt-2">
            <textarea
              value={chatInput}
              onChange={(e) => setChatInput(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendChat(); } }}
              placeholder="Ask about a customer, segment, or intervention… (Enter to send, Shift+Enter for new line)"
              rows={4}
              className="flex-1 rounded-xl border-2 border-[#818CF8] bg-white px-4 py-3 text-[14px] text-[#1E1B4B] resize-none focus:outline-none focus:border-[#4F46E5] focus:shadow-[0_0_0_4px_rgba(99,102,241,0.18)]"
            />
            <button
              onClick={sendChat}
              disabled={!chatInput.trim() || chatLoading}
              className="px-6 py-3 rounded-xl font-bold text-[14px] text-white disabled:opacity-50 self-end transition-all hover:-translate-y-0.5"
              style={{ background: "linear-gradient(135deg, #6366F1, #4338CA)", boxShadow: "0 4px 16px rgba(99,102,241,0.35)" }}
            >
              Send
            </button>
          </div>
          <p className="text-[12px] text-[#9CA3AF] mt-1.5">The AI uses real data from Supabase. It calls multiple tools before answering — responses take 10–20 seconds.</p>
          {chatMessages.length > 0 && (
            <button onClick={() => setChatMessages([])} className="text-[13px] text-[#EF4444] font-semibold mt-1">
              Clear conversation
            </button>
          )}
        </div>
      )}
    </div>
  );
}
