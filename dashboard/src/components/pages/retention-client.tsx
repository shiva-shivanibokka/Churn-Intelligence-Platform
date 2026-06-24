"use client";

import { useMemo, useState } from "react";
import { Customer } from "@/lib/supabase";
import { PageTitle, SectionHeading } from "@/components/ui/section-heading";
import { MetricCard } from "@/components/ui/metric-card";

interface Props { customers: Customer[] }

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
  trace?: unknown[];
};

export function RetentionClient({ customers }: Props) {
  const [selectedId, setSelectedId] = useState("");
  const [loading, setLoading] = useState(false);
  const [action, setAction] = useState<Action | null>(null);
  const [chatMessages, setChatMessages] = useState<{ role: string; content: string }[]>([]);
  const [chatInput, setChatInput] = useState("");
  const [chatLoading, setChatLoading] = useState(false);
  const [tab, setTab] = useState<"batch" | "chat">("batch");

  const persuadables = useMemo(
    () => customers.filter((c) => c.customer_type === "Persuadable").sort((a, b) => b.net_roi - a.net_roi),
    [customers]
  );

  const selected = useMemo(() => customers.find((c) => c.customer_id === selectedId), [customers, selectedId]);

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
    const userMsg = { role: "user", content: chatInput };
    const history = [...chatMessages, userMsg];
    setChatMessages(history);
    setChatInput("");
    setChatLoading(true);
    try {
      const res = await fetch("/api/agent", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mode: "chat", message: chatInput, history: chatMessages }),
      });
      const data = await res.json();
      setChatMessages([...history, { role: "assistant", content: data.response ?? data.error ?? "No response." }]);
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
                {action.trace && (action.trace as unknown[]).length > 0 && (
                  <div className="mt-3">
                    <p className="text-[12px] font-bold uppercase tracking-wide text-[#7C3AED] mb-2">
                      Agent Reasoning — {(action.trace as unknown[]).length} tool calls
                    </p>
                    <div className="space-y-2">
                      {(action.trace as { round: number; tool: string; args: Record<string, unknown>; result: unknown }[]).map((step, si) => (
                        <div key={si} className="bg-[#F5F3FF] border border-[#DDD6FE] rounded-xl px-4 py-3">
                          <div className="flex items-center gap-2 mb-1">
                            <span className="w-5 h-5 rounded-full bg-[#6366F1] text-white text-[10px] font-bold flex items-center justify-center shrink-0">{step.round}</span>
                            <span className="text-[13px] font-semibold text-[#4338CA]">{step.tool}</span>
                          </div>
                          <p className="text-[11px] text-[#6B7280] mb-1">Args: {JSON.stringify(step.args)}</p>
                          <p className="text-[11px] text-[#1E1B4B] bg-white rounded-lg px-2 py-1 border border-[#E0E7FF] overflow-x-auto whitespace-pre-wrap">
                            {typeof step.result === "string" ? step.result : JSON.stringify(step.result, null, 1)}
                          </p>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
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
