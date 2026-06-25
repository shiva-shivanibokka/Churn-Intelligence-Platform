"use client";

import { useState, useCallback, useMemo } from "react";
import { RetentionAction } from "@/lib/supabase";
import { saveFeedback } from "@/lib/data";
import { PageTitle, SectionHeading } from "@/components/ui/section-heading";
import { MetricCard } from "@/components/ui/metric-card";

interface Props {
  actions: RetentionAction[];
}

function computeSummary(actions: RetentionAction[]) {
  const retained = actions.filter((a) => a.outcome === "retained").length;
  const churned  = actions.filter((a) => a.outcome === "churned").length;
  const total    = actions.length;

  const byTypeMap: Record<string, { total: number; retained: number; withFeedback: number }> = {};
  const bySegMap:  Record<string, { total: number; retained: number; withFeedback: number }> = {};

  for (const a of actions) {
    const t = a.intervention_type ?? "Unknown";
    const s = a.segment ?? "Unknown";
    if (!byTypeMap[t]) byTypeMap[t] = { total: 0, retained: 0, withFeedback: 0 };
    if (!bySegMap[s])  bySegMap[s]  = { total: 0, retained: 0, withFeedback: 0 };
    byTypeMap[t].total++;
    bySegMap[s].total++;
    if (a.outcome) {
      byTypeMap[t].withFeedback++;
      bySegMap[s].withFeedback++;
      if (a.outcome === "retained") { byTypeMap[t].retained++; bySegMap[s].retained++; }
    }
  }

  return {
    total, retained, churned,
    pending: total - retained - churned,
    byType: Object.entries(byTypeMap).map(([type, v]) => ({ type, ...v, rate: v.withFeedback ? Math.round((v.retained / v.withFeedback) * 100) : null })),
    bySeg:  Object.entries(bySegMap).map(([segment, v]) => ({ segment, ...v, rate: v.withFeedback ? Math.round((v.retained / v.withFeedback) * 100) : null })),
  };
}

export function AnalyticsClient({ actions: initialActions }: Props) {
  const [actions, setActions] = useState(initialActions);
  const [saving, setSaving] = useState<string | null>(null);

  const summary = useMemo(() => computeSummary(actions), [actions]);

  const updateOutcome = useCallback(async (action: RetentionAction, outcome: string) => {
    if (saving) return;
    setSaving(action.id);
    try {
      await saveFeedback(action.id, action.customer_id, outcome);
      setActions((prev) => prev.map((a) => a.id === action.id ? { ...a, outcome } : a));
    } catch {
      // silent — user can retry
    }
    setSaving(null);
  }, [saving]);

  const retentionRate = summary.retained + summary.churned > 0
    ? Math.round((summary.retained / (summary.retained + summary.churned)) * 100)
    : null;

  return (
    <div>
      <PageTitle>Audit & Analytics</PageTitle>

      {/* Campaign Summary */}
      <SectionHeading>Campaign Summary</SectionHeading>
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        <MetricCard label="Total Actions" value={summary.total.toLocaleString()} accentColor="#4F46E5" />
        <MetricCard label="Retained" value={summary.retained.toLocaleString()} delta={summary.total ? `${Math.round(summary.retained / summary.total * 100)}% of total` : "—"} accentColor="#10B981" />
        <MetricCard label="Churned" value={summary.churned.toLocaleString()} accentColor="#EF4444" />
        <MetricCard label="Overall Retention Rate" value={retentionRate !== null ? `${retentionRate}%` : "—"} delta="of customers with feedback" accentColor="#7C3AED" />
      </div>

      {/* By intervention type */}
      <SectionHeading>Retention Rate by Intervention Type</SectionHeading>
      <div className="bg-white rounded-2xl border-2 border-[#DDD6FE] overflow-hidden shadow-sm mb-8">
        <table className="w-full text-[14px]">
          <thead>
            <tr style={{ background: "linear-gradient(110deg, #4338CA 0%, #7C3AED 100%)" }}>
              {["Intervention Type", "Total Actions", "With Feedback", "Retention Rate"].map((h) => (
                <th key={h} className="text-white font-bold text-left px-4 py-3 text-[12px] uppercase tracking-wide">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {summary.byType.length === 0 && (
              <tr>
                <td colSpan={4} className="px-4 py-6 text-center text-[#A78BFA] text-[14px]">
                  No retention actions yet. Generate plans on the Retention Actions page.
                </td>
              </tr>
            )}
            {summary.byType.map((r, i) => (
              <tr key={r.type} className={i % 2 === 0 ? "bg-white" : "bg-[#F5F3FF]"}>
                <td className="px-4 py-3 font-semibold">{r.type ?? "Unknown"}</td>
                <td className="px-4 py-3">{r.total}</td>
                <td className="px-4 py-3">{r.withFeedback}</td>
                <td className="px-4 py-3">
                  {r.rate !== null ? (
                    <span className={`font-bold ${r.rate >= 70 ? "text-[#10B981]" : r.rate >= 40 ? "text-[#F59E0B]" : "text-[#EF4444]"}`}>
                      {r.rate}%
                    </span>
                  ) : <span className="text-[#A78BFA] italic">No feedback yet</span>}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* By segment */}
      <SectionHeading>Retention Rate by Segment</SectionHeading>
      <div className="bg-white rounded-2xl border-2 border-[#DDD6FE] overflow-hidden shadow-sm mb-8">
        <table className="w-full text-[14px]">
          <thead>
            <tr style={{ background: "linear-gradient(110deg, #4338CA 0%, #7C3AED 100%)" }}>
              {["Segment", "Total Actions", "With Feedback", "Retention Rate"].map((h) => (
                <th key={h} className="text-white font-bold text-left px-4 py-3 text-[12px] uppercase tracking-wide">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {summary.bySeg.length === 0 && (
              <tr>
                <td colSpan={4} className="px-4 py-6 text-center text-[#A78BFA] text-[14px]">No data yet.</td>
              </tr>
            )}
            {summary.bySeg.map((r, i) => (
              <tr key={r.segment} className={i % 2 === 0 ? "bg-white" : "bg-[#F5F3FF]"}>
                <td className="px-4 py-3 font-semibold">{r.segment ?? "Unknown"}</td>
                <td className="px-4 py-3">{r.total}</td>
                <td className="px-4 py-3">{r.withFeedback}</td>
                <td className="px-4 py-3">
                  {r.rate !== null ? (
                    <span className={`font-bold ${r.rate >= 70 ? "text-[#10B981]" : r.rate >= 40 ? "text-[#F59E0B]" : "text-[#EF4444]"}`}>
                      {r.rate}%
                    </span>
                  ) : <span className="text-[#A78BFA] italic">No feedback yet</span>}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Full action log */}
      <SectionHeading>Full Action Log</SectionHeading>
      <div className="bg-white rounded-2xl border-2 border-[#DDD6FE] overflow-hidden shadow-sm overflow-x-auto">
        <table className="w-full text-[13px] min-w-[800px]">
          <thead>
            <tr style={{ background: "linear-gradient(110deg, #4338CA 0%, #7C3AED 100%)" }}>
              {["Customer", "Segment", "Churn Prob", "Intervention", "Channel", "Mode", "Outcome", "Update", "Generated"].map((h) => (
                <th key={h} className="text-white font-bold text-left px-3 py-3 text-[11px] uppercase tracking-wide">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {actions.length === 0 && (
              <tr>
                <td colSpan={9} className="px-4 py-8 text-center text-[#A78BFA]">No actions generated yet.</td>
              </tr>
            )}
            {actions.map((a, i) => (
              <tr key={a.id} className={i % 2 === 0 ? "bg-white" : "bg-[#F5F3FF]"}>
                <td className="px-3 py-2.5 font-mono text-[#4F46E5] font-semibold">{a.customer_id}</td>
                <td className="px-3 py-2.5">{a.segment ?? "—"}</td>
                <td className="px-3 py-2.5">{a.churn_probability !== null ? `${(a.churn_probability * 100).toFixed(1)}%` : "—"}</td>
                <td className="px-3 py-2.5">{a.intervention_type ?? "—"}</td>
                <td className="px-3 py-2.5">{a.channel ?? "—"}</td>
                <td className="px-3 py-2.5">{a.agentic_mode ? "Agentic" : "Standard"}</td>
                <td className="px-3 py-2.5">
                  <span className={`px-2 py-0.5 rounded-full text-[11px] font-bold ${
                    a.outcome === "retained" ? "bg-[#D1FAE5] text-[#065F46]" :
                    a.outcome === "churned" ? "bg-[#FEE2E2] text-[#991B1B]" :
                    "bg-[#F3F4F6] text-[#6B7280]"
                  }`}>
                    {a.outcome ?? "pending"}
                  </span>
                </td>
                <td className="px-3 py-2.5">
                  <div className="flex gap-1">
                    <button
                      onClick={() => updateOutcome(a, "retained")}
                      disabled={saving === a.id || a.outcome === "retained"}
                      className="px-2 py-0.5 rounded-full text-[10px] font-bold border transition-all disabled:opacity-40"
                      style={{ borderColor: "#10B981", color: a.outcome === "retained" ? "white" : "#10B981", background: a.outcome === "retained" ? "#10B981" : "transparent" }}
                      title="Mark as retained"
                    >
                      ✓ Retained
                    </button>
                    <button
                      onClick={() => updateOutcome(a, "churned")}
                      disabled={saving === a.id || a.outcome === "churned"}
                      className="px-2 py-0.5 rounded-full text-[10px] font-bold border transition-all disabled:opacity-40"
                      style={{ borderColor: "#EF4444", color: a.outcome === "churned" ? "white" : "#EF4444", background: a.outcome === "churned" ? "#EF4444" : "transparent" }}
                      title="Mark as churned"
                    >
                      ✗ Churned
                    </button>
                  </div>
                </td>
                <td className="px-3 py-2.5 text-[#9CA3AF]">{new Date(a.generated_at).toLocaleDateString()}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
