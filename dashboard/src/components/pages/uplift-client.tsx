"use client";

import { useMemo } from "react";
import dynamic from "next/dynamic";
import { Customer } from "@/lib/supabase";
import { PageTitle, SectionHeading } from "@/components/ui/section-heading";
import { MetricCard } from "@/components/ui/metric-card";
import { ChartCard } from "@/components/ui/chart-card";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell } from "recharts";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

const TYPE_COLORS: Record<string, string> = {
  "Persuadable":  "#6366F1",
  "Sure Thing":   "#10B981",
  "Lost Cause":   "#F43F5E",
  "Sleeping Dog": "#F59E0B",
};

const TYPE_DESCRIPTIONS: Record<string, string> = {
  "Persuadable":  "High churn risk + responds well to intervention → your #1 targeting priority",
  "Sure Thing":   "Would stay anyway — intervention is wasted spend",
  "Lost Cause":   "Already decided to leave — intervention won't help",
  "Sleeping Dog": "Low churn risk but intervention might actually trigger them to leave — leave alone",
};

interface Props { customers: Customer[] }

export function UpliftClient({ customers }: Props) {
  const kpis = useMemo(() => {
    const persuadable = customers.filter((c) => c.customer_type === "Persuadable").length;
    const positiveROI = customers.filter((c) => c.roi_positive).length;
    const avgUplift = customers.reduce((s, c) => s + c.uplift_score, 0) / customers.length;
    const totalROI = customers.filter((c) => c.roi_positive).reduce((s, c) => s + c.net_roi, 0);
    return { persuadable, positiveROI, avgUplift, totalROI };
  }, [customers]);

  // Uplift vs ChurnProb scatter by customer type
  const byType = useMemo(() => {
    const types = [...new Set(customers.map((c) => c.customer_type))];
    return types.map((t) => {
      const rows = customers.filter((c) => c.customer_type === t);
      return {
        type: "scatter" as const,
        mode: "markers" as const,
        name: t,
        x: rows.map((c) => c.churn_probability),
        y: rows.map((c) => c.uplift_score),
        text: rows.map((c) => `${c.customer_id}<br>Segment: ${c.segment}<br>ROI: $${c.net_roi.toFixed(0)}`),
        marker: { size: 7, color: TYPE_COLORS[t] ?? "#6B7280", opacity: 0.8, line: { width: 0.5, color: "white" } },
        hovertemplate: "%{text}<extra>%{fullData.name}</extra>",
      };
    });
  }, [customers]);

  // ROI by segment
  const roiBySegment = useMemo(() => {
    const segs = [...new Set(customers.map((c) => c.segment))];
    return segs.map((seg) => {
      const rows = customers.filter((c) => c.segment === seg && c.customer_type === "Persuadable");
      return {
        segment: seg,
        avgROI: rows.length ? rows.reduce((s, c) => s + c.net_roi, 0) / rows.length : 0,
        count: rows.length,
      };
    }).sort((a, b) => b.avgROI - a.avgROI);
  }, [customers]);

  // Customer type distribution
  const typeDistribution = useMemo(() => {
    const types = [...new Set(customers.map((c) => c.customer_type))];
    return types.map((t) => ({
      type: t,
      count: customers.filter((c) => c.customer_type === t).length,
      color: TYPE_COLORS[t] ?? "#6B7280",
    }));
  }, [customers]);

  // Top Persuadables table
  const topPersuadables = useMemo(() =>
    customers
      .filter((c) => c.customer_type === "Persuadable")
      .sort((a, b) => b.net_roi - a.net_roi)
      .slice(0, 15),
    [customers]
  );

  return (
    <div>
      <PageTitle>Uplift Intelligence</PageTitle>

      {/* Plain-English primer */}
      <div className="bg-[#EEF2FF] border-l-4 border-[#6366F1] rounded-r-xl px-4 py-3 mb-4 text-[14px] text-[#1E1B4B]">
        <strong>What is uplift modelling?</strong> Standard churn models tell you <em>who will churn</em>. Uplift models tell you <em>who will change their mind if you intervene</em> — which is a completely different question. Someone who is 90% likely to churn but would leave regardless of your offer is not worth spending money on. Someone who is 60% likely to churn but would stay with the right intervention is worth a lot.
      </div>
      <div className="bg-[#F0FDF4] border-l-4 border-[#10B981] rounded-r-xl px-4 py-3 mb-6 text-[14px] text-[#14532D]">
        <strong>The T–S Learner:</strong> We trained two separate XGBoost models — one on customers who received a past intervention, one on those who did not. Uplift Score = P(stay | intervene) − P(stay | no intervention). <strong>Positive uplift</strong> = intervention helps. <strong>Negative uplift</strong> = intervention backfires. The four customer types below are derived from this score.
      </div>

      {/* Customer type explainer cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 mb-8">
        {Object.entries(TYPE_DESCRIPTIONS).map(([type, desc]) => (
          <div key={type} className="rounded-2xl border-2 p-4 bg-white shadow-sm" style={{ borderColor: TYPE_COLORS[type] ?? "#E0E7FF" }}>
            <div className="text-[13px] font-bold mb-1" style={{ color: TYPE_COLORS[type] }}>{type}</div>
            <div className="text-[12px] text-[#6B7280] leading-snug">{desc}</div>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        <MetricCard label="Persuadables" value={kpis.persuadable.toLocaleString()} delta="Worth targeting" accentColor="#6366F1" />
        <MetricCard label="Positive ROI Interventions" value={kpis.positiveROI.toLocaleString()} delta="Financially justified" accentColor="#10B981" />
        <MetricCard label="Avg Uplift Score" value={kpis.avgUplift >= 0 ? `+${(kpis.avgUplift * 100).toFixed(2)}%` : `${(kpis.avgUplift * 100).toFixed(2)}%`} accentColor="#A855F7" />
        <MetricCard label="Total ROI Potential" value={`$${(kpis.totalROI / 1000).toFixed(0)}k`} delta="From Persuadables only" accentColor="#F59E0B" />
      </div>

      {/* Uplift quadrant scatter */}
      <SectionHeading>Uplift Quadrant — Who to Target</SectionHeading>
      <div className="bg-[#EEF2FF] border border-[#DDD6FE] rounded-xl px-4 py-2.5 mb-3 text-[13px] text-[#4338CA]">
        X-axis = churn probability (how likely they are to leave). Y-axis = uplift score (how much an intervention helps). The <strong>top-right quadrant</strong> (high churn risk + high uplift) contains your Persuadables — the only customers worth spending budget on. Dashed lines show the thresholds. Don't waste spend on the top-left (Sure Things stay anyway) or bottom-right (Sleeping Dogs: intervening makes them more likely to leave).
      </div>
      <ChartCard>
        <Plot
          data={byType as Plotly.Data[]}
          layout={{
            height: 540,
            template: "plotly_white" as Plotly.Template,
            xaxis: { title: "Churn Probability →", zeroline: true, range: [0, 1] },
            yaxis: { title: "Uplift Score →", zeroline: true },
            shapes: [
              { type: "line" as const, x0: 0.3, x1: 0.3, y0: -1, y1: 1, line: { dash: "dot", color: "#9CA3AF", width: 1.5 } },
              { type: "line" as const, x0: 0, x1: 1, y0: 0.05, y1: 0.05, line: { dash: "dot", color: "#9CA3AF", width: 1.5 } },
            ],
            annotations: [
              { x: 0.65, y: 0.5, text: "Persuadables<br>← Target here", showarrow: false, font: { color: "#6366F1", size: 13, family: "Inter" } },
              { x: 0.15, y: 0.5, text: "Sure Things", showarrow: false, font: { color: "#10B981", size: 12, family: "Inter" } },
              { x: 0.65, y: -0.4, text: "Sleeping Dogs", showarrow: false, font: { color: "#F59E0B", size: 12, family: "Inter" } },
              { x: 0.15, y: -0.4, text: "Lost Causes", showarrow: false, font: { color: "#F43F5E", size: 12, family: "Inter" } },
            ],
            margin: { l: 60, r: 30, t: 30, b: 60 },
            legend: { orientation: "h", y: 1.07, x: 0, font: { size: 13 } },
            paper_bgcolor: "white",
            plot_bgcolor: "#FAFAFA",
            font: { family: "Inter, sans-serif", color: "#334155" },
          }}
          config={{ responsive: true, displayModeBar: true }}
          style={{ width: "100%" }}
          useResizeHandler
        />
      </ChartCard>

      <div className="h-8" />

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Customer type distribution */}
        <div>
          <SectionHeading>Customer Type Distribution</SectionHeading>
          <div className="bg-[#EEF2FF] border border-[#DDD6FE] rounded-xl px-4 py-2.5 mb-3 text-[13px] text-[#4338CA]">
            How many customers fall into each of the four uplift categories. The <span className="font-semibold text-[#6366F1]">Persuadables</span> bar is your total addressable campaign audience.
          </div>
          <ChartCard>
            <ResponsiveContainer width="100%" height={340}>
              <BarChart data={typeDistribution} margin={{ top: 10, right: 20, left: 0, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#E0E7FF" />
                <XAxis dataKey="type" tick={{ fontSize: 12, fill: "#6B7280" }} />
                <YAxis tick={{ fontSize: 12, fill: "#6B7280" }} />
                <Tooltip contentStyle={{ borderRadius: "10px", border: "2px solid #DDD6FE", fontSize: 13 }} />
                <Bar dataKey="count" name="Customers" radius={[5, 5, 0, 0]}>
                  {typeDistribution.map((d, i) => <Cell key={i} fill={d.color} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </ChartCard>
        </div>

        {/* ROI by segment */}
        <div>
          <SectionHeading>Expected ROI per Intervention by Segment</SectionHeading>
          <div className="bg-[#EEF2FF] border border-[#DDD6FE] rounded-xl px-4 py-2.5 mb-3 text-[13px] text-[#4338CA]">
            Average net dollar return per intervention, calculated only for Persuadables in each segment. Longer bar = more financially attractive segment to run a campaign against first.
          </div>
          <ChartCard>
            <ResponsiveContainer width="100%" height={340}>
              <BarChart data={roiBySegment} layout="vertical" margin={{ top: 10, right: 30, left: 120, bottom: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#E0E7FF" />
                <XAxis type="number" tickFormatter={(v) => `$${v.toFixed(0)}`} tick={{ fontSize: 11, fill: "#6B7280" }} />
                <YAxis type="category" dataKey="segment" tick={{ fontSize: 12, fill: "#1E1B4B" }} width={115} />
                <Tooltip contentStyle={{ borderRadius: "10px", border: "2px solid #DDD6FE", fontSize: 13 }} formatter={(v) => [`$${Number(v).toFixed(2)}`, "Avg ROI"]} />
                <Bar dataKey="avgROI" radius={[0, 5, 5, 0]} name="Avg ROI">
                  {roiBySegment.map((_, i) => (
                    <Cell key={i} fill={["#6366F1", "#A855F7", "#06B6D4", "#10B981", "#F59E0B"][i % 5]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </ChartCard>
        </div>
      </div>

      <div className="h-8" />

      {/* Top Persuadables table */}
      <SectionHeading>Top 15 Persuadables by Net ROI — Your Campaign Priority List</SectionHeading>
      <div className="bg-[#EEF2FF] border border-[#DDD6FE] rounded-xl px-4 py-2.5 mb-3 text-[13px] text-[#4338CA]">
        These are the customers where a retention intervention is both <em>most likely to work</em> (positive uplift score) and <em>most financially valuable</em> (highest net ROI after intervention cost). Run your next campaign starting from the top of this list — it maximises return per dollar spent.
      </div>
      <div className="bg-white rounded-2xl border-2 border-[#DDD6FE] overflow-hidden shadow-sm">
        <table className="w-full text-[14px]">
          <thead>
            <tr style={{ background: "linear-gradient(110deg, #6366F1 0%, #A855F7 100%)" }}>
              {["#", "Customer ID", "Segment", "Churn Prob", "Uplift Score", "Net ROI", "Priority Rank"].map((h) => (
                <th key={h} className="text-white font-bold text-left px-4 py-3 text-[12px] uppercase tracking-wide">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {topPersuadables.map((c, i) => (
              <tr key={c.customer_id} className={i % 2 === 0 ? "bg-white" : "bg-[#F5F3FF]"}>
                <td className="px-4 py-2.5 text-[#9CA3AF] font-semibold">{i + 1}</td>
                <td className="px-4 py-2.5 font-mono text-[13px] text-[#6366F1] font-semibold">{c.customer_id}</td>
                <td className="px-4 py-2.5">{c.segment}</td>
                <td className="px-4 py-2.5 font-semibold text-[#F43F5E]">{(c.churn_probability * 100).toFixed(1)}%</td>
                <td className="px-4 py-2.5 font-semibold text-[#10B981]">{c.uplift_score >= 0 ? "+" : ""}{(c.uplift_score * 100).toFixed(2)}%</td>
                <td className="px-4 py-2.5 font-bold text-[#6366F1]">${c.net_roi.toFixed(2)}</td>
                <td className="px-4 py-2.5 text-[#6B7280]">{c.intervention_priority ?? "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
