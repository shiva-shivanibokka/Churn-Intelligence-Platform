"use client";

import { useMemo, useState } from "react";
import dynamic from "next/dynamic";
import { Customer } from "@/lib/supabase";
import { PageTitle, SectionHeading } from "@/components/ui/section-heading";
import { MetricCard } from "@/components/ui/metric-card";
import { ChartCard } from "@/components/ui/chart-card";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

const SEGMENT_COLORS: Record<string, string> = {
  "Champions":       "#6366F1",
  "Loyal Customers": "#A855F7",
  "At-Risk":         "#F43F5E",
  "Hibernating":     "#F59E0B",
  "Lost Customers":  "#64748B",
};

const COLOR_OPTIONS = ["Segment", "Churn", "RiskTier", "ChurnProbability", "UpliftScore"];

const UMAP_CAPTIONS: Record<string, { label: string; caption: string }> = {
  Segment: {
    label: "Coloured by Segment",
    caption: "Each dot is one customer. Well-separated colour clusters confirm the 5 segments are behaviourally distinct — customers within a cluster behave similarly to each other and differently from other clusters. This is the foundation of targeted retention.",
  },
  Churn: {
    label: "Coloured by Actual Churn (0/1)",
    caption: "Red = customer actually churned, green = stayed. Notice churners cluster in specific regions of the map — this validates that UMAP preserved the churn signal, and the model is learning real patterns rather than noise.",
  },
  RiskTier: {
    label: "Coloured by Predicted Risk Tier",
    caption: "Shows where High / Medium / Low risk customers sit in behavioural space. High-Risk customers (red) should overlap heavily with churner clusters above — if they do, the model's risk tier cutoffs are well-calibrated.",
  },
  ChurnProbability: {
    label: "Coloured by Predicted Churn Probability",
    caption: "Gradient from green (safe) to red (likely to churn). Dense red zones are where the model is most confident about churn — these are your highest-priority outreach targets. Scattered red dots in green zones are customers the model is uncertain about.",
  },
  UpliftScore: {
    label: "Coloured by Uplift Score",
    caption: "Uplift = how much an intervention would reduce churn probability for that specific customer. Green = high uplift (intervention helps a lot), red = intervention would backfire. Targeting green customers in red risk zones gives you the highest ROI.",
  },
};

interface Props { customers: Customer[] }

export function SegmentationClient({ customers }: Props) {
  const [colorBy, setColorBy] = useState("Segment");

  const segments = useMemo(() => {
    const map: Record<string, Customer[]> = {};
    for (const c of customers) {
      if (!map[c.segment]) map[c.segment] = [];
      map[c.segment].push(c);
    }
    return map;
  }, [customers]);

  const kpiData = useMemo(() =>
    Object.entries(segments).map(([seg, rows]) => ({
      segment: seg,
      count: rows.length,
      churnRate: rows.filter((r) => r.churn === 1).length / rows.length,
      color: SEGMENT_COLORS[seg] ?? "#6B7280",
    })), [segments]);

  const umapTraces = useMemo(() => {
    if (colorBy === "Segment") {
      return Object.entries(segments).map(([seg, rows]) => ({
        type: "scatter" as const,
        mode: "markers" as const,
        name: seg,
        x: rows.map((r) => r.umap_1),
        y: rows.map((r) => r.umap_2),
        text: rows.map((r) => `Customer ${r.customer_id}<br>Seg: ${r.segment}<br>Churn Prob: ${(r.churn_probability * 100).toFixed(1)}%`),
        marker: { size: 8, color: SEGMENT_COLORS[seg] ?? "#6B7280", opacity: 0.82, line: { width: 0.5, color: "white" } },
        hovertemplate: "%{text}<extra>%{fullData.name}</extra>",
      }));
    }

    const colorValues = customers.map((c) => {
      if (colorBy === "Churn") return c.churn;
      if (colorBy === "ChurnProbability") return c.churn_probability;
      if (colorBy === "UpliftScore") return c.uplift_score;
      if (colorBy === "RiskTier") return c.risk_tier === "High Risk" ? 1 : c.risk_tier === "Medium Risk" ? 0.5 : 0;
      return 0;
    });

    return [{
      type: "scatter" as const,
      mode: "markers" as const,
      name: colorBy,
      x: customers.map((c) => c.umap_1),
      y: customers.map((c) => c.umap_2),
      marker: {
        size: 8,
        color: colorValues,
        colorscale: colorBy === "UpliftScore" ? "RdYlGn" : "RdYlGn_r",
        showscale: true,
        opacity: 0.82,
        line: { width: 0.5, color: "white" },
      },
      text: customers.map((c) => `Customer ${c.customer_id}<br>Seg: ${c.segment}`),
      hovertemplate: "%{text}<extra></extra>",
    }];
  }, [colorBy, customers, segments]);

  const gmmData = useMemo(() => {
    return Object.keys(segments).map((seg) => {
      const rows = segments[seg];
      const confs = rows.map((r) => {
        const probs = [r.gmm_prob_seg0, r.gmm_prob_seg1, r.gmm_prob_seg2, r.gmm_prob_seg3, r.gmm_prob_seg4]
          .filter((v): v is number => v !== null);
        return probs.length ? Math.max(...probs) : 1;
      });
      return {
        segment: seg,
        "High ≥90%":    confs.filter((v) => v >= 0.9).length,
        "Medium 80-90%": confs.filter((v) => v >= 0.8 && v < 0.9).length,
        "Boundary <80%": confs.filter((v) => v < 0.8).length,
      };
    });
  }, [segments]);

  const heatmapData = useMemo(() => {
    const features = ["tenure", "satisfaction_score", "days_since_last_order", "hour_spend_on_app", "cashback_amount"];
    const labels = ["Tenure", "Satisfaction", "Days Since Order", "App Hours", "Cashback"];
    const segs = Object.keys(segments);
    return { labels, segs, values: features.map((f) =>
      segs.map((s) => {
        const vals = segments[s].map((r) => (r as Record<string, unknown>)[f] as number).filter((v) => v != null);
        return vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : 0;
      })
    )};
  }, [segments]);

  const currentCaption = UMAP_CAPTIONS[colorBy] ?? UMAP_CAPTIONS.Segment;

  return (
    <div>
      <PageTitle>Customer Segmentation</PageTitle>

      <div className="bg-[#EEF2FF] border-l-4 border-[#6366F1] rounded-r-xl px-4 py-3 mb-6 text-[14px] text-[#1E1B4B]">
        <strong>What this page shows:</strong> Your ~1,500 customers have been grouped into 5 behavioural segments using K-Means clustering on purchase, engagement, and satisfaction patterns. Each segment needs a different retention strategy. Use the UMAP to visually explore how similar customers cluster together.
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-4 mb-8">
        {kpiData.map((k) => (
          <MetricCard key={k.segment} label={k.segment} value={k.count.toLocaleString()} delta={`${(k.churnRate * 100).toFixed(1)}% churn rate`} accentColor={k.color} />
        ))}
      </div>

      {/* UMAP */}
      <SectionHeading>Customer Behavioural Space (UMAP 2D Projection)</SectionHeading>
      <div className="flex items-center gap-3 mb-3">
        <label className="text-[13px] font-semibold text-[#6366F1]">Colour by:</label>
        <select
          value={colorBy}
          onChange={(e) => setColorBy(e.target.value)}
          className="rounded-xl border-2 border-[#818CF8] bg-white px-3 py-2 text-[14px] text-[#1E1B4B] font-medium min-w-[200px] focus:outline-none focus:border-[#6366F1]"
        >
          {COLOR_OPTIONS.map((o) => <option key={o}>{o}</option>)}
        </select>
        <span className="text-[13px] text-[#7C3AED] font-medium">{currentCaption.label}</span>
      </div>
      <div className="bg-[#F5F3FF] border border-[#DDD6FE] rounded-xl px-4 py-2.5 mb-3 text-[13px] text-[#4338CA]">
        {currentCaption.caption}
      </div>
      <ChartCard>
        <Plot
          data={umapTraces as Plotly.Data[]}
          layout={{
            height: 620,
            template: "plotly_white" as Plotly.Template,
            margin: { l: 30, r: 30, t: 20, b: 30 },
            legend: { orientation: "h", y: 1.04, x: 0, font: { size: 13 } },
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

      {/* Heatmap */}
      <SectionHeading>Segment Feature Heatmap</SectionHeading>
      <div className="bg-[#F5F3FF] border border-[#DDD6FE] rounded-xl px-4 py-2.5 mb-3 text-[13px] text-[#4338CA]">
        Average value of each feature per segment. This is how you interpret each cluster: Champions have high tenure and cashback; At-Risk customers have high days-since-last-order and low satisfaction. Use this to write segment-specific messaging.
      </div>
      <ChartCard>
        <Plot
          data={heatmapData.values.map((row, fi) => ({
            type: "bar" as const,
            name: heatmapData.labels[fi],
            x: heatmapData.segs,
            y: row,
          }))}
          layout={{
            height: 460,
            barmode: "group" as const,
            colorway: ["#6366F1", "#A855F7", "#F43F5E", "#F59E0B", "#06B6D4"],
            template: "plotly_white" as Plotly.Template,
            margin: { l: 40, r: 20, t: 20, b: 80 },
            legend: { orientation: "h", y: -0.25, font: { size: 13 } },
            paper_bgcolor: "white",
            plot_bgcolor: "#FAFAFA",
            font: { family: "Inter, sans-serif", color: "#334155" },
          }}
          config={{ responsive: true }}
          style={{ width: "100%" }}
          useResizeHandler
        />
      </ChartCard>

      <div className="h-8" />

      {/* GMM Confidence */}
      <SectionHeading>Segment Assignment Confidence</SectionHeading>
      <div className="bg-[#F5F3FF] border border-[#DDD6FE] rounded-xl px-4 py-2.5 mb-3 text-[13px] text-[#4338CA]">
        After K-Means assigns each customer to a segment, Gaussian Mixture Models (GMM) score how <em>confident</em> that assignment is. <strong>Indigo = clearly belongs (≥90%)</strong>, amber = sits between two segments, red = borderline and warrants manual review before targeting.
      </div>
      <ChartCard>
        <ResponsiveContainer width="100%" height={420}>
          <BarChart data={gmmData} margin={{ top: 10, right: 20, left: 0, bottom: 20 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#E0E7FF" />
            <XAxis dataKey="segment" tick={{ fontSize: 13, fill: "#6B7280" }} />
            <YAxis tick={{ fontSize: 12, fill: "#6B7280" }} />
            <Tooltip contentStyle={{ borderRadius: "10px", border: "2px solid #DDD6FE", fontSize: 13 }} />
            <Legend wrapperStyle={{ fontSize: 13, paddingTop: 12 }} />
            <Bar dataKey="High ≥90%"     stackId="a" fill="#6366F1" />
            <Bar dataKey="Medium 80-90%" stackId="a" fill="#F59E0B" />
            <Bar dataKey="Boundary <80%" stackId="a" fill="#F43F5E" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </ChartCard>

      <div className="h-8" />

      {/* Summary table */}
      <SectionHeading>Segment Summary Table</SectionHeading>
      <div className="bg-[#F5F3FF] border border-[#DDD6FE] rounded-xl px-4 py-2.5 mb-3 text-[13px] text-[#4338CA]">
        Quick reference: how big each segment is, how many are actually churning, and what share are classified as Persuadable (worth targeting with a retention campaign).
      </div>
      <div className="bg-white rounded-2xl border-2 border-[#DDD6FE] overflow-hidden shadow-sm">
        <table className="w-full text-[14px]">
          <thead>
            <tr style={{ background: "linear-gradient(110deg, #6366F1 0%, #A855F7 100%)" }}>
              {["Segment", "Customers", "Churn Rate", "Avg Churn Prob", "High Risk %", "Persuadable %"].map((h) => (
                <th key={h} className="text-white font-bold text-left px-4 py-3 text-[12px] uppercase tracking-wide">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {kpiData.map((k, i) => {
              const rows = segments[k.segment];
              const avgProb = rows.reduce((s, r) => s + r.churn_probability, 0) / rows.length;
              const highRisk = rows.filter((r) => r.risk_tier === "High Risk").length / rows.length;
              const persuadable = rows.filter((r) => r.customer_type === "Persuadable").length / rows.length;
              return (
                <tr key={k.segment} className={i % 2 === 0 ? "bg-white" : "bg-[#F5F3FF]"}>
                  <td className="px-4 py-3 font-semibold flex items-center gap-2">
                    <span className="w-3 h-3 rounded-full shrink-0" style={{ background: k.color }} />
                    {k.segment}
                  </td>
                  <td className="px-4 py-3">{k.count.toLocaleString()}</td>
                  <td className="px-4 py-3 font-semibold" style={{ color: k.churnRate > 0.3 ? "#F43F5E" : "#10B981" }}>
                    {(k.churnRate * 100).toFixed(1)}%
                  </td>
                  <td className="px-4 py-3">{(avgProb * 100).toFixed(1)}%</td>
                  <td className="px-4 py-3">{(highRisk * 100).toFixed(1)}%</td>
                  <td className="px-4 py-3 font-semibold text-[#6366F1]">{(persuadable * 100).toFixed(1)}%</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
