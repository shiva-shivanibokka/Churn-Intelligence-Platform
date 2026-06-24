"use client";

import { useMemo, useState } from "react";
import { Customer } from "@/lib/supabase";
import { PageTitle, SectionHeading } from "@/components/ui/section-heading";
import { MetricCard } from "@/components/ui/metric-card";
import { ChartCard } from "@/components/ui/chart-card";
import {
  BarChart, Bar, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from "recharts";

const SEGMENT_COLORS = ["#6366F1", "#A855F7", "#F43F5E", "#F59E0B", "#06B6D4"];
const TIER_COLORS: Record<string, string> = { "High Risk": "#F43F5E", "Medium Risk": "#F59E0B", "Low Risk": "#10B981" };

interface Props { customers: Customer[] }

export function ChurnClient({ customers }: Props) {
  const [segFilter, setSegFilter] = useState<string | null>(null);

  const segments = useMemo(() => [...new Set(customers.map((c) => c.segment))].sort(), [customers]);

  const filtered = useMemo(
    () => (segFilter ? customers.filter((c) => c.segment === segFilter) : customers),
    [customers, segFilter]
  );

  const kpis = useMemo(() => {
    const highRisk = filtered.filter((c) => c.risk_tier === "High Risk").length;
    const avgProb = filtered.reduce((s, c) => s + c.churn_probability, 0) / (filtered.length || 1);
    const actualChurners = filtered.filter((c) => c.churn === 1).length;
    return { highRisk, avgProb, actualChurners, total: filtered.length };
  }, [filtered]);

  // Probability histogram (buckets 0–10%, 10–20%, … 90–100%)
  const probHist = useMemo(() => {
    const buckets = Array.from({ length: 10 }, (_, i) => ({
      range: `${i * 10}–${(i + 1) * 10}%`,
      count: 0,
    }));
    for (const c of filtered) {
      const idx = Math.min(Math.floor(c.churn_probability * 10), 9);
      buckets[idx].count++;
    }
    return buckets;
  }, [filtered]);

  // Risk tier grouped by segment
  const tierBySeg = useMemo(() => {
    const segs = segments;
    return segs.map((seg, si) => {
      const rows = customers.filter((c) => c.segment === seg);
      return {
        segment: seg,
        "High Risk":   rows.filter((c) => c.risk_tier === "High Risk").length,
        "Medium Risk": rows.filter((c) => c.risk_tier === "Medium Risk").length,
        "Low Risk":    rows.filter((c) => c.risk_tier === "Low Risk").length,
        color: SEGMENT_COLORS[si % SEGMENT_COLORS.length],
      };
    });
  }, [customers, segments]);

  // SHAP bar — top features by avg |SHAP| for the filtered set
  const shapData = useMemo(() => {
    const totals: Record<string, { sum: number; count: number }> = {};
    for (const c of filtered) {
      const shap = (c.top_shap_features as Record<string, number>) ?? {};
      for (const [feat, val] of Object.entries(shap)) {
        if (!totals[feat]) totals[feat] = { sum: 0, count: 0 };
        totals[feat].sum += Math.abs(val);
        totals[feat].count++;
      }
    }
    return Object.entries(totals)
      .map(([feature, { sum, count }]) => ({ feature, importance: sum / count }))
      .sort((a, b) => b.importance - a.importance)
      .slice(0, 8);
  }, [filtered]);

  const histColors = probHist.map((_, i) => {
    if (i >= 7) return "#F43F5E";
    if (i >= 4) return "#F59E0B";
    return "#10B981";
  });

  return (
    <div>
      <PageTitle>Churn Risk Dashboard</PageTitle>

      <div className="bg-[#FFF1F2] border-l-4 border-[#F43F5E] rounded-r-xl px-4 py-3 mb-6 text-[14px] text-[#7F1D1D]">
        <strong>What this page shows:</strong> The XGBoost churn model scored every customer with a probability of churning (0–100%). This page breaks that down by segment, risk tier, and the specific features (SHAP values) driving each score. Use the <strong>segment filter below</strong> to zoom into one group — all four charts update together to reflect only those customers.
      </div>

      {/* Segment filter chips */}
      <div className="flex flex-wrap items-center gap-2 mb-6">
        <span className="text-[13px] font-semibold text-[#6B7280] mr-1">Filter by segment:</span>
        <button
          onClick={() => setSegFilter(null)}
          className="px-3 py-1.5 rounded-full text-[13px] font-semibold border-2 transition-all"
          style={!segFilter
            ? { background: "#6366F1", borderColor: "#6366F1", color: "white" }
            : { background: "white", borderColor: "#DDD6FE", color: "#6366F1" }}
        >
          All Segments
        </button>
        {segments.map((s, i) => (
          <button
            key={s}
            onClick={() => setSegFilter(segFilter === s ? null : s)}
            className="px-3 py-1.5 rounded-full text-[13px] font-semibold border-2 transition-all"
            style={segFilter === s
              ? { background: SEGMENT_COLORS[i % SEGMENT_COLORS.length], borderColor: SEGMENT_COLORS[i % SEGMENT_COLORS.length], color: "white" }
              : { background: "white", borderColor: "#DDD6FE", color: SEGMENT_COLORS[i % SEGMENT_COLORS.length] }}
          >
            {s}
          </button>
        ))}
        {segFilter && (
          <span className="text-[12px] text-[#6B7280] ml-2 italic">
            Showing {kpis.total.toLocaleString()} customers in <strong>{segFilter}</strong> — all charts reflect this filter.
          </span>
        )}
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        <MetricCard label="Total Customers" value={kpis.total.toLocaleString()} accentColor="#6366F1" />
        <MetricCard label="High Risk" value={kpis.highRisk.toLocaleString()} delta={`${((kpis.highRisk / kpis.total) * 100).toFixed(1)}% of group`} accentColor="#F43F5E" />
        <MetricCard label="Avg Churn Prob" value={`${(kpis.avgProb * 100).toFixed(1)}%`} accentColor="#F59E0B" />
        <MetricCard label="Actual Churners" value={kpis.actualChurners.toLocaleString()} delta={`${((kpis.actualChurners / kpis.total) * 100).toFixed(1)}% observed`} accentColor="#A855F7" />
      </div>

      {/* Probability distribution */}
      <SectionHeading>Churn Probability Distribution</SectionHeading>
      <div className="bg-[#FFF1F2] border border-[#FECDD3] rounded-xl px-4 py-2.5 mb-3 text-[13px] text-[#9F1239]">
        How many customers fall into each 10%-wide risk bucket. <span className="font-semibold text-[#10B981]">Green bars (0–40%)</span> = low risk, light-touch nurture. <span className="font-semibold text-[#F59E0B]">Amber (40–70%)</span> = medium risk, monitor closely. <span className="font-semibold text-[#F43F5E]">Red (70–100%)</span> = high risk, immediate intervention needed before they cancel.
      </div>
      <ChartCard>
        <ResponsiveContainer width="100%" height={420}>
          <BarChart data={probHist} margin={{ top: 10, right: 20, left: 0, bottom: 30 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#FFE4E6" />
            <XAxis dataKey="range" tick={{ fontSize: 12, fill: "#6B7280" }} label={{ value: "Churn Probability Bucket", position: "insideBottom", offset: -15, fontSize: 13, fill: "#9CA3AF" }} />
            <YAxis tick={{ fontSize: 12, fill: "#6B7280" }} />
            <Tooltip contentStyle={{ borderRadius: "10px", border: "2px solid #FECDD3", fontSize: 13 }} formatter={(v) => [v, "Customers"]} />
            <Bar dataKey="count" name="Customers" radius={[5, 5, 0, 0]}>
              {probHist.map((_, i) => <Cell key={i} fill={histColors[i]} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </ChartCard>

      <div className="h-8" />

      {/* Risk tier by segment */}
      <SectionHeading>Risk Tier Breakdown by Segment</SectionHeading>
      <div className="bg-[#FFF1F2] border border-[#FECDD3] rounded-xl px-4 py-2.5 mb-3 text-[13px] text-[#9F1239]">
        Stacked bars show the absolute number of High / Medium / Low risk customers in each segment. A segment dominated by <span className="font-semibold text-[#F43F5E]">red</span> needs an urgent campaign; mostly <span className="font-semibold text-[#10B981]">green</span> only needs light-touch nurture. This always shows all segments regardless of the segment filter above, so you can compare them side by side.
      </div>
      <ChartCard>
        <ResponsiveContainer width="100%" height={400}>
          <BarChart data={tierBySeg} margin={{ top: 10, right: 20, left: 0, bottom: 30 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#FFE4E6" />
            <XAxis dataKey="segment" tick={{ fontSize: 12, fill: "#1E1B4B" }} />
            <YAxis tick={{ fontSize: 12, fill: "#6B7280" }} />
            <Tooltip contentStyle={{ borderRadius: "10px", border: "2px solid #FECDD3", fontSize: 13 }} />
            <Legend wrapperStyle={{ fontSize: 13, paddingTop: 12 }} />
            <Bar dataKey="High Risk"   stackId="a" fill={TIER_COLORS["High Risk"]} />
            <Bar dataKey="Medium Risk" stackId="a" fill={TIER_COLORS["Medium Risk"]} />
            <Bar dataKey="Low Risk"    stackId="a" fill={TIER_COLORS["Low Risk"]} radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </ChartCard>

      <div className="h-8" />

      {/* SHAP feature importance */}
      <SectionHeading>Top Churn Drivers — SHAP Feature Importance</SectionHeading>
      <div className="bg-[#EEF2FF] border border-[#DDD6FE] rounded-xl px-4 py-2.5 mb-3 text-[13px] text-[#4338CA]">
        SHAP (SHapley Additive exPlanations) measures exactly how much each feature pushes the churn probability up or down for each customer. This shows the average importance across {segFilter ? `the ${segFilter} segment` : "all customers"}. <strong>Longer bar = bigger influence on whether someone churns.</strong> These are the levers your retention campaigns should pull first — target the top driver with your most impactful offer.
      </div>
      <ChartCard>
        <ResponsiveContainer width="100%" height={400}>
          <BarChart data={shapData} layout="vertical" margin={{ top: 10, right: 30, left: 130, bottom: 10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#E0E7FF" />
            <XAxis type="number" tick={{ fontSize: 11, fill: "#6B7280" }} tickFormatter={(v) => v.toFixed(3)} />
            <YAxis type="category" dataKey="feature" tick={{ fontSize: 12, fill: "#1E1B4B" }} width={125} />
            <Tooltip contentStyle={{ borderRadius: "10px", border: "2px solid #DDD6FE", fontSize: 13 }} formatter={(v) => [Number(v).toFixed(4), "Avg |SHAP|"]} />
            <Bar dataKey="importance" name="Importance" radius={[0, 5, 5, 0]}>
              {shapData.map((_, i) => (
                <Cell key={i} fill={
                  i === 0 ? "#F43F5E" : i === 1 ? "#F97316" : i === 2 ? "#F59E0B" :
                  i === 3 ? "#6366F1" : i === 4 ? "#A855F7" : "#06B6D4"
                } />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </ChartCard>

      <div className="h-8" />

      {/* Avg churn prob by segment */}

      <SectionHeading>Average Churn Probability by Segment</SectionHeading>
      <div className="bg-[#EEF2FF] border border-[#DDD6FE] rounded-xl px-4 py-2.5 mb-3 text-[13px] text-[#4338CA]">
        Compares the mean predicted churn probability across all 5 segments. Higher bar = higher urgency for that group. Use this to prioritise which segment to run your next retention campaign against.
      </div>
      <ChartCard>
        <ResponsiveContainer width="100%" height={340}>
          <BarChart
            data={segments.map((seg, i) => ({
              segment: seg,
              avgProb: customers.filter((c) => c.segment === seg).reduce((s, c) => s + c.churn_probability, 0) /
                (customers.filter((c) => c.segment === seg).length || 1),
              color: SEGMENT_COLORS[i % SEGMENT_COLORS.length],
            }))}
            layout="vertical"
            margin={{ top: 10, right: 40, left: 130, bottom: 10 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#E0E7FF" />
            <XAxis type="number" tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} tick={{ fontSize: 11, fill: "#6B7280" }} />
            <YAxis type="category" dataKey="segment" tick={{ fontSize: 12, fill: "#1E1B4B" }} width={125} />
            <Tooltip contentStyle={{ borderRadius: "10px", border: "2px solid #DDD6FE", fontSize: 13 }} formatter={(v) => [`${(Number(v) * 100).toFixed(1)}%`, "Avg Churn Prob"]} />
            <Bar dataKey="avgProb" name="Avg Churn Prob" radius={[0, 5, 5, 0]}>
              {segments.map((_, i) => <Cell key={i} fill={SEGMENT_COLORS[i % SEGMENT_COLORS.length]} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </ChartCard>

      {/* Glossary */}
      <div className="mt-8 bg-[#F8FAFC] border border-[#E2E8F0] rounded-2xl p-5">
        <p className="text-[12px] font-bold uppercase tracking-wide text-[#64748B] mb-3">Parameter Glossary</p>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-[13px]">
          {[
            ["Churn Probability", "The XGBoost model's predicted probability (0–100%) that a customer will leave. Each segment has its own model trained separately."],
            ["Risk Tier", "Churn probability bucketed into three tiers: Low Risk (0–30%), Medium Risk (30–60%), High Risk (60–100%)."],
            ["SHAP Value", "SHapley Additive exPlanations — measures how much each feature pushes the churn probability up or down for an individual customer. Positive = increases churn risk."],
            ["Segment Filter", "Selecting a segment updates the probability histogram and SHAP chart to show only those customers. Use this to understand what drives churn within one group specifically."],
            ["Complain", "Whether a customer has filed a complaint (1=yes, 0=no). One of the strongest churn predictors — unresolved complaints are a leading indicator of cancellation."],
            ["SatisfactionScore", "Customer-reported satisfaction (1=best, 5=worst). Inverted in the SupportRiskScore composite feature."],
          ].map(([term, def]) => (
            <div key={term} className="flex gap-2">
              <span className="font-semibold text-[#4338CA] shrink-0">{term}:</span>
              <span className="text-[#475569]">{def}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
