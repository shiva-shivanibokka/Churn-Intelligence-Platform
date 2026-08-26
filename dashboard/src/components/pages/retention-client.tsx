"use client";

import { useCallback, useMemo, useState, useSyncExternalStore } from "react";
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
  // Set when the plan was generated but could not be written to
  // retention_actions. The plan on screen is real; the audit trail is not.
  save_error?: string;
};

type TraceStep = {
  round: number;
  tool: string;
  args: Record<string, unknown>;
  result: Record<string, unknown>;
};

type ChatMessage = { role: string; content: string; trace?: TraceStep[] };

const TOOL_META: Record<string, { label: string; color: string }> = {
  get_top_churn_drivers:          { label: "Top Churn Drivers",          color: "#F43F5E" },
  lookup_customer_details:        { label: "Customer Details",           color: "#6366F1" },
  get_segment_benchmark:          { label: "Segment Benchmark",          color: "#A855F7" },
  get_all_segment_benchmarks:     { label: "All Segment Benchmarks",     color: "#A855F7" },
  search_retention_playbook:      { label: "Retention Playbook",         color: "#F59E0B" },
  calculate_intervention_roi:     { label: "ROI Calculation",            color: "#10B981" },
  get_past_interventions:         { label: "Past Interventions",         color: "#0EA5E9" },
  get_intervention_success_rates: { label: "Success Rates by Type",      color: "#0EA5E9" },
  get_at_risk_customers:          { label: "At-Risk Customers",          color: "#EF4444" },
  get_revenue_at_risk:            { label: "Revenue at Risk",            color: "#F97316" },
  save_retention_action:          { label: "Save Action to Database",    color: "#10B981" },
  get_unactioned_persuadables:    { label: "Unactioned Persuadables",    color: "#8B5CF6" },
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
        <p className="text-[#1E1B4B] italic mt-1">&ldquo;{String(result.message ?? "")}&rdquo;</p>
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
  if (tool === "get_past_interventions") {
    if (result.message) return <p className="text-[12px] text-[#0EA5E9] italic">{String(result.message)}</p>;
    const rows = Array.isArray(result) ? result as Record<string, unknown>[] : [];
    return (
      <div className="space-y-2">
        {rows.map((r, i) => (
          <div key={i} className="flex items-center gap-3 text-[12px]">
            <span className={`px-2 py-0.5 rounded-full text-[10px] font-bold shrink-0 ${r.outcome === "retained" ? "bg-[#D1FAE5] text-[#065F46]" : r.outcome === "churned" ? "bg-[#FEE2E2] text-[#991B1B]" : "bg-[#F3F4F6] text-[#6B7280]"}`}>
              {String(r.outcome)}
            </span>
            <span className="font-semibold text-[#1E1B4B]">{String(r.intervention_type ?? "—")}</span>
            <span className="text-[#6B7280]">via {String(r.channel ?? "—")}</span>
            <span className="text-[#9CA3AF] ml-auto shrink-0">{r.generated_at ? new Date(String(r.generated_at)).toLocaleDateString() : "—"}</span>
          </div>
        ))}
      </div>
    );
  }
  if (tool === "get_intervention_success_rates") {
    const rows = Array.isArray(result) ? result as Record<string, unknown>[] : [];
    return (
      <div className="overflow-x-auto">
        <table className="text-[11px] w-full border-collapse">
          <thead><tr className="bg-[#EFF6FF]">
            {["Intervention Type", "Total", "With Feedback", "Retention Rate"].map((h) => (
              <th key={h} className="text-left px-2 py-1 text-[#0EA5E9] font-bold border-b border-[#BFDBFE]">{h}</th>
            ))}
          </tr></thead>
          <tbody>{rows.map((r) => {
            const rate = String(r.retention_rate ?? "—");
            const rateNum = parseInt(rate);
            return (
              <tr key={String(r.intervention_type)} className="border-b border-[#F0F9FF]">
                <td className="px-2 py-1 font-semibold">{String(r.intervention_type)}</td>
                <td className="px-2 py-1">{String(r.total_actions)}</td>
                <td className="px-2 py-1">{String(r.with_feedback)}</td>
                <td className={`px-2 py-1 font-bold ${!isNaN(rateNum) ? (rateNum >= 70 ? "text-[#10B981]" : rateNum >= 40 ? "text-[#F59E0B]" : "text-[#EF4444]") : "text-[#9CA3AF] italic"}`}>
                  {rate}
                </td>
              </tr>
            );
          })}</tbody>
        </table>
      </div>
    );
  }
  if (tool === "get_at_risk_customers") {
    if (result.message) return <p className="text-[12px] text-[#EF4444] italic">{String(result.message)}</p>;
    const rows = Array.isArray(result) ? result as Record<string, unknown>[] : [];
    return (
      <div className="overflow-x-auto">
        <table className="text-[11px] w-full border-collapse">
          <thead><tr className="bg-[#FEF2F2]">
            {["Customer ID", "Segment", "Churn Prob", "Type", "Uplift", "Net ROI"].map((h) => (
              <th key={h} className="text-left px-2 py-1 text-[#EF4444] font-bold border-b border-[#FECACA]">{h}</th>
            ))}
          </tr></thead>
          <tbody>{rows.map((r) => (
            <tr key={String(r.customer_id)} className="border-b border-[#FFF5F5]">
              <td className="px-2 py-1 font-mono text-[#4F46E5]">{String(r.customer_id)}</td>
              <td className="px-2 py-1">{String(r.segment)}</td>
              <td className="px-2 py-1 font-bold text-[#EF4444]">{String(r.churn_probability)}</td>
              <td className="px-2 py-1">{String(r.customer_type)}</td>
              <td className="px-2 py-1">{String(r.uplift_score)}</td>
              <td className="px-2 py-1 text-[#10B981]">{String(r.net_roi)}</td>
            </tr>
          ))}</tbody>
        </table>
      </div>
    );
  }
  if (tool === "get_revenue_at_risk") {
    const total = String(result.total_revenue_at_risk ?? "—");
    const bySegment = Array.isArray(result.by_segment) ? result.by_segment as Record<string, unknown>[] : [];
    return (
      <div className="space-y-2">
        <div className="flex gap-6 text-[12px]">
          <div><span className="text-[#F97316] font-semibold">Total at Risk: </span><span className="font-bold text-[14px]">{total}</span></div>
          {result.total_expected_churners !== undefined && (
            <div><span className="text-[#F97316] font-semibold">Expected Churners: </span><span className="font-bold">{String(result.total_expected_churners)}</span></div>
          )}
        </div>
        {bySegment.length > 0 && (
          <div className="space-y-1">
            {bySegment.map((s) => (
              <div key={String(s.segment)} className="flex items-center gap-2 text-[11px]">
                <span className="w-36 font-semibold text-[#1E1B4B] shrink-0">{String(s.segment)}</span>
                <span className="text-[#6B7280]">{String(s.expected_churners)} churners</span>
                <span className="ml-auto font-bold text-[#F97316]">{String(s.revenue_at_risk)}</span>
              </div>
            ))}
          </div>
        )}
        {result.note != null && <p className="text-[10px] text-[#9CA3AF] italic">{String(result.note)}</p>}
      </div>
    );
  }
  if (tool === "save_retention_action") {
    const ok = result.success as boolean;
    return (
      <div className={`text-[12px] rounded-lg px-3 py-2 ${ok ? "bg-[#D1FAE5] text-[#065F46]" : "bg-[#FEE2E2] text-[#991B1B]"}`}>
        <p className="font-bold">{ok ? "✓ Saved" : "✗ Failed"}</p>
        <p>{String((result.message ?? result.error) ?? "")}</p>
        {result.action_id != null && <p className="text-[10px] mt-1 opacity-70">ID: {String(result.action_id)}</p>}
      </div>
    );
  }
  if (tool === "get_unactioned_persuadables") {
    if (result.message) return <p className="text-[12px] text-[#8B5CF6] italic">{String(result.message)}</p>;
    const rows = Array.isArray(result) ? result as Record<string, unknown>[] : [];
    return (
      <div className="overflow-x-auto">
        <table className="text-[11px] w-full border-collapse">
          <thead><tr className="bg-[#F5F3FF]">
            {["Customer ID", "Segment", "Churn Prob", "Uplift", "Net ROI"].map((h) => (
              <th key={h} className="text-left px-2 py-1 text-[#7C3AED] font-bold border-b border-[#EDE9FE]">{h}</th>
            ))}
          </tr></thead>
          <tbody>{rows.map((r) => (
            <tr key={String(r.customer_id)} className="border-b border-[#F3F0FF]">
              <td className="px-2 py-1 font-mono text-[#4F46E5]">{String(r.customer_id)}</td>
              <td className="px-2 py-1">{String(r.segment)}</td>
              <td className="px-2 py-1 font-bold text-[#EF4444]">{String(r.churn_probability)}</td>
              <td className="px-2 py-1">{String(r.uplift_score)}</td>
              <td className="px-2 py-1 font-bold text-[#10B981]">{String(r.net_roi)}</td>
            </tr>
          ))}</tbody>
        </table>
      </div>
    );
  }
  return <pre className="text-[11px] text-[#6B7280] whitespace-pre-wrap">{JSON.stringify(result, null, 2)}</pre>;
}

/**
 * Just enough Markdown for what this agent actually emits.
 *
 * Asked to compare five segments, the model replies with a pipe table and bold
 * headers — which was being rendered through `whitespace-pre-wrap`, so visitors
 * read `| **Lapsed** | **33.3 %** |` as literal text. The answer was good and
 * looked like debug output.
 *
 * This covers headings, bold, bullets and tables, which is the whole vocabulary
 * in the recorded runs. It is not a Markdown parser and does not want to be —
 * `react-markdown` plus a sanitiser is a real dependency for a formatting
 * problem this size, and the input is one model's output in one component.
 *
 * Nothing here uses `dangerouslySetInnerHTML`. The text comes from a language
 * model — with tool results from the database woven into it — so it is built
 * into React elements, where it cannot become markup.
 */
function inline(text: string, keyPrefix: string) {
  // Split on the delimiter rather than matching pairs: odd indices are the
  // emphasised runs, which is true even when a stray `**` leaves one unclosed.
  return text.split("**").map((part, i) =>
    i % 2 === 1
      ? <strong key={`${keyPrefix}-${i}`} className="font-bold text-[#1E1B4B]">{part}</strong>
      : <span key={`${keyPrefix}-${i}`}>{part}</span>
  );
}

function AgentMarkdown({ text }: { text: string }) {
  const lines = (text ?? "").split("\n");
  const blocks: React.ReactNode[] = [];
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];

    // Table: a header row, a separator of dashes, then body rows.
    if (line.trim().startsWith("|") && lines[i + 1]?.includes("---")) {
      const cells = (row: string) =>
        row.split("|").slice(1, -1).map((c) => c.trim());
      const header = cells(line);
      const rows: string[][] = [];
      i += 2;
      while (i < lines.length && lines[i].trim().startsWith("|")) {
        rows.push(cells(lines[i]));
        i++;
      }
      blocks.push(
        <div key={`t-${i}`} className="overflow-x-auto my-3">
          <table className="w-full text-[13px] border border-[#DDD6FE] rounded-lg overflow-hidden">
            <thead>
              <tr className="bg-[#F5F3FF]">
                {header.map((h, hi) => (
                  <th key={hi} className="text-left px-3 py-2 font-bold text-[#4F46E5] whitespace-nowrap">
                    {inline(h, `h${hi}`)}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((row, ri) => (
                <tr key={ri} className="border-t border-[#EDE9FE]">
                  {row.map((cell, ci) => (
                    <td key={ci} className="px-3 py-2 text-[#1E1B4B] tabular-nums">
                      {inline(cell, `c${ri}-${ci}`)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      );
      continue;
    }

    if (line.startsWith("#")) {
      const heading = line.replace(/^#+\s*/, "");
      blocks.push(
        <p key={`h-${i}`} className="text-[14px] font-bold text-[#1E1B4B] mt-3 mb-1">
          {inline(heading, `hd${i}`)}
        </p>
      );
      i++;
      continue;
    }

    if (/^\s*[-*]\s+/.test(line)) {
      const items: string[] = [];
      while (i < lines.length && /^\s*[-*]\s+/.test(lines[i])) {
        items.push(lines[i].replace(/^\s*[-*]\s+/, ""));
        i++;
      }
      blocks.push(
        <ul key={`u-${i}`} className="list-disc pl-5 my-2 space-y-1">
          {items.map((item, ii) => (
            <li key={ii} className="text-[14px] text-[#1E1B4B] leading-relaxed">
              {inline(item, `li${ii}`)}
            </li>
          ))}
        </ul>
      );
      continue;
    }

    if (!line.trim()) {
      i++;
      continue;
    }

    // Consecutive non-blank lines are one paragraph.
    const para: string[] = [];
    while (
      i < lines.length &&
      lines[i].trim() &&
      !lines[i].startsWith("#") &&
      !lines[i].trim().startsWith("|") &&
      !/^\s*[-*]\s+/.test(lines[i])
    ) {
      para.push(lines[i]);
      i++;
    }
    blocks.push(
      <p key={`p-${i}`} className="text-[14px] text-[#1E1B4B] leading-relaxed my-2">
        {inline(para.join(" "), `pp${i}`)}
      </p>
    );
  }

  return <div>{blocks}</div>;
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

/**
 * Where the visitor supplies their own Groq key.
 *
 * This deployment holds no Groq credential at all. A public agent endpoint
 * backed by the owner's free tier is a 100k-token-a-day budget any visitor can
 * finish, and once finished the feature is dead for everyone until the next
 * reset — dead in a way that reads as "this project is broken". So the agent
 * runs on the visitor's own key, which also means this project has no secret to
 * leak, rotate, or pay for.
 *
 * The key is held in sessionStorage, not localStorage: it survives navigating
 * between the dashboard's pages and disappears when the tab closes, which is
 * the right lifetime for a credential someone pasted into a demo. It is sent as
 * a request header and never as a query parameter — a key in a URL ends up in
 * server access logs and browser history, where nobody thinks to clear it.
 */
const KEY_STORAGE = "churn-engine.groq-key";

/**
 * sessionStorage as an external store, read through useSyncExternalStore.
 *
 * The obvious approach — `useState("")` plus an effect that reads storage on
 * mount — sets state during an effect and triggers a second render pass on
 * every load. Reading it in a lazy initialiser instead is worse: the server
 * renders an empty key and the client renders the stored one, which is a
 * hydration mismatch.
 *
 * This is precisely what useSyncExternalStore is for. `getServerSnapshot`
 * returns the empty string, so SSR and the first client paint agree, and React
 * resubscribes to the real value immediately afterwards.
 */
const keyStore = {
  listeners: new Set<() => void>(),

  read(): string {
    try {
      return window.sessionStorage.getItem(KEY_STORAGE) ?? "";
    } catch {
      // Private browsing and blocked storage: the panel still works, the key
      // just does not survive navigating between pages.
      return "";
    }
  },

  write(value: string) {
    try {
      if (value) window.sessionStorage.setItem(KEY_STORAGE, value);
      else window.sessionStorage.removeItem(KEY_STORAGE);
    } catch {
      // Non-fatal — see above.
    }
    for (const listener of this.listeners) listener();
  },

  subscribe(listener: () => void) {
    keyStore.listeners.add(listener);
    // Another tab clearing the key should clear it here too.
    window.addEventListener("storage", listener);
    return () => {
      keyStore.listeners.delete(listener);
      window.removeEventListener("storage", listener);
    };
  },
};

function useGroqKey(): [string, (key: string) => void] {
  const key = useSyncExternalStore(
    keyStore.subscribe,
    () => keyStore.read(),
    () => "",
  );
  const setKey = useCallback((value: string) => keyStore.write(value), []);
  return [key, setKey];
}

function ApiKeyPanel({
  apiKey,
  onChange,
  invalid,
}: {
  apiKey: string;
  onChange: (key: string) => void;
  invalid: boolean;
}) {
  const [draft, setDraft] = useState("");
  const [open, setOpen] = useState(false);

  const configured = apiKey.length > 0;
  const masked = configured ? `gsk_••••••••${apiKey.slice(-4)}` : "";

  // Collapsed once a key is in place: it is setup, not a control the visitor
  // needs in front of them while working.
  const expanded = open || !configured || invalid;

  return (
    <div
      className={`rounded-2xl border-2 mb-6 overflow-hidden ${
        invalid ? "border-[#FECACA] bg-[#FEF2F2]" : configured ? "border-[#DDD6FE] bg-white" : "border-[#FDE68A] bg-[#FFFBEB]"
      }`}
    >
      <button
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-center justify-between gap-3 px-5 py-3.5 text-left"
        aria-expanded={expanded}
      >
        <span className="flex items-center gap-2.5 text-[14px] font-bold text-[#1E1B4B]">
          <span aria-hidden="true">{configured && !invalid ? "🔑" : "⚠️"}</span>
          {configured && !invalid
            ? <>Using your Groq key <span className="font-mono font-normal text-[13px] text-[#6B7280]">{masked}</span></>
            : "The AI agent needs your Groq API key"}
        </span>
        <span className="text-[12px] font-semibold text-[#6366F1] shrink-0">
          {expanded ? "Hide" : "Change"}
        </span>
      </button>

      {expanded && (
        <div className="px-5 pb-5 pt-1">
          <p className="text-[13px] text-[#4B5563] leading-relaxed mb-3">
            This demo deliberately ships without a Groq key of its own. A shared free
            tier is 100,000 tokens a day for every visitor combined, so the first
            person to hold down the button turns the agent off for everyone else.
            Bringing your own means the agent is never queued behind a stranger.{" "}
            <a
              href="https://console.groq.com/keys"
              target="_blank"
              rel="noopener noreferrer"
              className="font-semibold text-[#4F46E5] underline underline-offset-2"
            >
              Groq keys are free
            </a>{" "}
            and take about a minute to create.
          </p>

          <p className="text-[12px] text-[#6B7280] leading-relaxed mb-3">
            Your key is kept in this browser tab only, sent as a request header to
            this site&rsquo;s own agent route, and forwarded to Groq. It is never
            stored on the server, never written to the database, never logged, and
            never placed in a URL. Close the tab and it is gone. The other four
            dashboard pages need no key at all.
          </p>

          <div className="flex gap-2 flex-wrap items-center">
            <input
              type="password"
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && draft.trim()) {
                  onChange(draft.trim());
                  setDraft("");
                  setOpen(false);
                }
              }}
              placeholder="gsk_…"
              autoComplete="off"
              spellCheck={false}
              aria-label="Groq API key"
              className="flex-1 min-w-[240px] rounded-xl border-2 border-[#818CF8] bg-white px-4 py-2.5 text-[14px] font-mono text-[#1E1B4B] focus:outline-none focus:border-[#4F46E5]"
            />
            <button
              onClick={() => { onChange(draft.trim()); setDraft(""); setOpen(false); }}
              disabled={!draft.trim()}
              className="px-5 py-2.5 rounded-xl font-bold text-[14px] text-white disabled:opacity-50 disabled:cursor-not-allowed"
              style={{ background: "linear-gradient(135deg, #6366F1, #4338CA)" }}
            >
              Use this key
            </button>
            {configured && (
              <button
                onClick={() => { onChange(""); setDraft(""); }}
                className="px-4 py-2.5 rounded-xl font-semibold text-[14px] text-[#6B7280] border-2 border-[#DDD6FE] bg-white hover:border-[#818CF8]"
              >
                Forget it
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

/**
 * The generated plan, its reasoning trace, and the failure states around them.
 *
 * Extracted so the replay tab renders a recorded run through exactly the same
 * markup as a live one. A replay that looks different from the thing it is
 * replaying is a second implementation to keep in step, and it invites the
 * reasonable suspicion that the recording is a mock-up rather than a real run.
 */
function PlanResult({ action, heading = "AI-Generated Retention Plan" }: { action: Action | null; heading?: string }) {
  const confidenceColor = (c?: string) =>
    c === "High" ? "#10B981" : c === "Medium" ? "#F59E0B" : "#EF4444";

  if (!action) return null;

  return (
    <>
          {!action.error && !action.do_not_intervene_reason && (
            <div>
              <SectionHeading>{heading}</SectionHeading>
              {action.save_error && (
                <div className="mb-4 rounded-xl border-2 border-[#FCD34D] bg-[#FFFBEB] px-4 py-3 text-[13px] text-[#92400E]">
                  <b>This plan was not recorded.</b> It is shown below and is
                  real, but the write to <code>retention_actions</code> failed,
                  so it will not appear in Audit &amp; Analytics.{" "}
                  {action.save_error}
                </div>
              )}
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

          {action.do_not_intervene_reason && (
            <div className="bg-[#FEF2F2] border-2 border-[#FECACA] rounded-2xl p-5">
              <p className="font-bold text-[#EF4444] mb-1">Do Not Intervene</p>
              <p className="text-[14px] text-[#7F1D1D]">{action.do_not_intervene_reason}</p>
            </div>
          )}

          {action.error && (
            <div className="bg-[#FEF2F2] border-2 border-[#FECACA] rounded-2xl p-5">
              <p className="font-bold text-[#EF4444]">Error: {action.error}</p>
            </div>
          )}
    </>
  );
}

/**
 * Recorded agent runs, for visitors without a Groq key.
 *
 * These are not mock-ups. Each one is a real call to this same `/api/agent`
 * route, made against the live Supabase data with a real key, captured by
 * `scripts/record_agent_runs.py` and committed as static JSON — the tool calls
 * happened, the numbers came out of the database, and the model wrote the
 * words. They render through the same components a live run does, so what a
 * visitor watches here is what they would get by pasting a key in.
 *
 * The alternative was leaving the agent behind a key prompt, which for anyone
 * evaluating the project in three minutes is indistinguishable from leaving it
 * broken.
 */
type ReplayIndexEntry = {
  id: string;
  mode: "batch" | "chat";
  label: string;
  tool_calls: number;
  elapsed_seconds: number;
};

type ReplayRun = {
  id: string;
  mode: "batch" | "chat";
  model: string;
  customer?: PersuadableCustomer;
  action?: Action | null;
  question?: string;
  response?: string;
  trace: TraceStep[];
  elapsed_seconds: number;
};

function ReplayTab() {
  const [index, setIndex] = useState<ReplayIndexEntry[] | null>(null);
  const [model, setModel] = useState("");
  const [selected, setSelected] = useState<ReplayRun | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [started, setStarted] = useState(false);

  const loadIndex = useCallback(async () => {
    try {
      const res = await fetch("/agent-runs/index.json");
      if (!res.ok) throw new Error(res.status + " " + res.statusText);
      const data = await res.json();
      setModel(data.model ?? "");
      setIndex(data.runs ?? []);
    } catch (e) {
      // Say the recordings are missing rather than rendering an empty picker,
      // which reads as "there is nothing to show" instead of "this failed".
      setLoadError(String(e));
      setIndex([]);
    }
  }, []);

  const openRun = useCallback(async (id: string) => {
    setBusy(true);
    setLoadError(null);
    try {
      const res = await fetch("/agent-runs/" + id + ".json");
      if (!res.ok) throw new Error(res.status + " " + res.statusText);
      setSelected(await res.json());
    } catch (e) {
      setLoadError(String(e));
      setSelected(null);
    }
    setBusy(false);
  }, []);

  if (!started) {
    return (
      <div className="rounded-2xl border-2 border-[#DDD6FE] bg-[#F5F3FF] px-5 py-6 text-center">
        <p className="text-[14px] text-[#4B5563] mb-4">
          Six agent runs were recorded against the live database and committed to
          this repository. No key required.
        </p>
        <button
          onClick={() => { setStarted(true); void loadIndex(); }}
          className="px-6 py-3 rounded-xl font-bold text-[14px] text-white"
          style={{ background: "linear-gradient(135deg, #6366F1, #4338CA)" }}
        >
          Load recorded runs
        </button>
      </div>
    );
  }

  if (index === null) {
    return <p className="text-[14px] text-[#6B7280]">Loading recorded runs…</p>;
  }

  return (
    <div className="space-y-6">
      <div className="rounded-2xl border-2 border-[#DDD6FE] bg-[#F5F3FF] px-5 py-4">
        <p className="text-[14px] font-bold text-[#1E1B4B] mb-1.5">
          Real runs, recorded — no key needed
        </p>
        <p className="text-[13px] text-[#4B5563] leading-relaxed">
          Every run below is an actual call to this dashboard&rsquo;s agent route, made
          against the live database{model ? <> with <span className="font-mono text-[12px]">{model}</span></> : null} and
          committed to the repository. The tool calls happened, the figures came out
          of Supabase, and the model wrote the text. Nothing here is a mock-up — it
          renders through the same components a live run does. Add your own key on
          the other two tabs to run fresh ones.
        </p>
      </div>

      {loadError && (
        <div className="rounded-2xl border-2 border-[#FECACA] bg-[#FEF2F2] px-5 py-4 text-[13px] text-[#7F1D1D]">
          <b>Could not load the recordings.</b> {loadError}
        </div>
      )}

      <div className="flex flex-wrap gap-2">
        {index.map((run) => (
          <button
            key={run.id}
            onClick={() => openRun(run.id)}
            className={
              "text-left px-4 py-3 rounded-xl border-2 transition-all max-w-[340px] " +
              (selected?.id === run.id
                ? "border-[#4F46E5] bg-white shadow-md"
                : "border-[#DDD6FE] bg-white hover:border-[#818CF8]")
            }
          >
            <span className="block text-[11px] font-bold uppercase tracking-wide text-[#7C3AED] mb-0.5">
              {run.mode === "batch" ? "Retention plan" : "Question"}
            </span>
            <span className="block text-[13px] font-semibold text-[#1E1B4B] leading-snug">
              {run.label}
            </span>
            <span className="block text-[11px] text-[#6B7280] mt-1 tabular-nums">
              {run.tool_calls} tool call{run.tool_calls === 1 ? "" : "s"} · {run.elapsed_seconds.toFixed(1)}s
            </span>
          </button>
        ))}
      </div>

      {busy && <p className="text-[14px] text-[#6B7280]">Loading run…</p>}

      {selected && !busy && (
        <div className="space-y-6">
          {selected.mode === "batch" && selected.customer && (
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
              <MetricCard label="Churn Probability" value={(selected.customer.churn_probability * 100).toFixed(1) + "%"} accentColor="#EF4444" />
              <MetricCard label="Uplift Score" value={(selected.customer.uplift_score >= 0 ? "+" : "") + (selected.customer.uplift_score * 100).toFixed(2) + "%"} accentColor="#4F46E5" />
              <MetricCard label="Net ROI" value={"$" + selected.customer.net_roi.toFixed(2)} accentColor="#10B981" />
              <MetricCard label="Customer Type" value={selected.customer.customer_type} accentColor="#7C3AED" />
            </div>
          )}

          {selected.mode === "batch" && (
            <PlanResult action={selected.action ?? null} heading="Recorded Retention Plan" />
          )}

          {selected.mode === "chat" && (
            <div>
              <SectionHeading>Recorded Conversation</SectionHeading>
              <div className="rounded-2xl border-2 border-[#DDD6FE] bg-white p-5 space-y-4">
                <div>
                  <p className="text-[11px] font-bold uppercase tracking-wide text-[#7C3AED] mb-1">Question</p>
                  <p className="text-[14px] text-[#1E1B4B]">{selected.question}</p>
                </div>
                <AgentTrace trace={selected.trace} defaultOpen />
                <div>
                  <p className="text-[11px] font-bold uppercase tracking-wide text-[#7C3AED] mb-1">Answer</p>
                  <AgentMarkdown text={selected.response ?? ""} />
                </div>
              </div>
            </div>
          )}
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
  const [tab, setTab] = useState<"batch" | "chat" | "replay">("replay");
  const [apiKey, setApiKey] = useGroqKey();
  const [keyRejected, setKeyRejected] = useState(false);

  const updateKey = useCallback((key: string) => {
    setApiKey(key);
    // A newly supplied key deserves a fresh verdict; leaving the rejection up
    // would keep the panel red after the visitor fixed the problem.
    setKeyRejected(false);
  }, [setApiKey]);

  /**
   * The agent route requires the visitor's own key, so it goes in a header.
   *
   * Not a query parameter and not the request body: headers are the one place a
   * credential does not get written to a server access log or left in the
   * browser's history.
   */
  const agentHeaders = useCallback((): HeadersInit => {
    const headers: Record<string, string> = { "Content-Type": "application/json" };
    if (apiKey) headers["x-groq-key"] = apiKey;
    return headers;
  }, [apiKey]);

  const selected = useMemo(() => persuadables.find((c) => c.customer_id === selectedId), [persuadables, selectedId]);

  async function generateAction() {
    if (!selected) return;
    setLoading(true);
    setAction(null);
    try {
      const res = await fetch("/api/agent", {
        method: "POST",
        headers: agentHeaders(),
        body: JSON.stringify({ mode: "batch", customer: selected }),
      });
      const data = await res.json();
      if (data.needs_key || data.invalid_key) setKeyRejected(true);
      const plan = data.action ?? { error: data.error };
      if (data.save_error) plan.save_error = data.save_error;
      setAction(plan);
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
        headers: agentHeaders(),
        body: JSON.stringify({ mode: "chat", message: chatInput, history: chatMessages.map(({ role, content }) => ({ role, content })) }),
      });
      const data = await res.json();
      if (data.needs_key || data.invalid_key) setKeyRejected(true);
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

  return (
    <div>
      <PageTitle>Retention Actions</PageTitle>

      {tab !== "replay" && (
        <ApiKeyPanel apiKey={apiKey} onChange={updateKey} invalid={keyRejected} />
      )}

      <div className="flex gap-2 mb-6">
        {(["replay", "batch", "chat"] as const).map((t) => (
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
            {t === "replay" ? "Watch a Recorded Run" : t === "batch" ? "Generate Action Plan" : "Ask AI Assistant"}
          </button>
        ))}
      </div>

      {tab === "replay" && <ReplayTab />}

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

          <PlanResult action={action} />
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
                {m.role === "assistant"
                  ? <AgentMarkdown text={m.content} />
                  : <p className="text-[#1E1B4B] whitespace-pre-wrap leading-relaxed">{m.content}</p>}
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
