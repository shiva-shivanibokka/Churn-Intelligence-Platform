"use client";

import { useMemo, useState } from "react";
import dynamic from "next/dynamic";
import { Customer } from "@/lib/supabase";
import { SegmentSummaryRow, UmapPoint } from "@/lib/data";
import { PageTitle, SectionHeading } from "@/components/ui/section-heading";
import { MetricCard } from "@/components/ui/metric-card";
import { ChartCard } from "@/components/ui/chart-card";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

// Actual segment names produced by segmentation.py
const SEGMENT_COLORS: Record<string, string> = {
  "Champions":      "#6366F1",
  "Loyal Customers":"#A855F7",
  "At-Risk":        "#F43F5E",
  "Price Sensitive":"#F59E0B",
  "Lapsed":         "#64748B",
};

// Segment definitions for the explainer panel
const SEGMENT_DEFINITIONS: Record<string, { tagline: string; detail: string }> = {
  "Champions": {
    tagline: "Highest engagement & lowest recency",
    detail: "Bought recently, order often, many devices registered — top-tier behaviorally. Note: Champions can still churn if they are discount-driven shoppers who leave when promotions stop.",
  },
  "Loyal Customers": {
    tagline: "Consistent behaviour, long tenure",
    detail: "Regular buyers with stable order counts and moderate satisfaction. Strong relationship, low urgency — light-touch nurture campaigns work well.",
  },
  "At-Risk": {
    tagline: "Engagement is declining",
    detail: "Were once active but days-since-last-order is rising and satisfaction is below average. High churn risk — prioritise with targeted win-back offers immediately.",
  },
  "Price Sensitive": {
    tagline: "Heavy coupon usage, discount-driven loyalty",
    detail: "Order frequently but mainly when discounts are active (high DiscountSensitivity score). Loyalty is conditional on price — they churn when competitors offer better deals.",
  },
  "Lapsed": {
    tagline: "Long since last purchase, low engagement",
    detail: "Highest days-since-last-order, lowest engagement scores. Many may have already churned mentally. Reactivation requires a compelling offer; some are not worth pursuing.",
  },
};

const COLOR_OPTIONS = ["Segment", "Churn", "RiskTier", "ChurnProbability", "UpliftScore"];

// Explicit hex colorscales — avoids Plotly.js named-scale rendering inconsistencies
const COLORSCALE_HIGH_BAD: Plotly.ColorScale = [[0, "#10B981"], [0.5, "#FCD34D"], [1, "#EF4444"]]; // green→amber→red
const COLORSCALE_HIGH_GOOD: Plotly.ColorScale = [[0, "#EF4444"], [0.5, "#FCD34D"], [1, "#10B981"]]; // red→amber→green

const UMAP_CAPTIONS: Record<string, { label: string; caption: string }> = {
  Segment: {
    label: "Coloured by Segment",
    caption: "Each dot is one customer (all ~1,500 are plotted). Well-separated colour blobs confirm the 5 segments are behaviourally distinct. Customers within a blob behave similarly; customers in different blobs behave differently. This is what makes targeted retention possible.",
  },
  Churn: {
    label: "Coloured by Actual Churn (0 = stayed, 1 = churned)",
    caption: "Green = customer stayed, Red = customer actually churned. Churners clustering in specific regions of the map validates that UMAP preserved the churn signal — the model is learning real patterns. Note: the colours are green and red (high = red using the green→amber→red scale).",
  },
  RiskTier: {
    label: "Coloured by Predicted Risk Tier",
    caption: "This view looks similar to Churn Probability because Risk Tier IS derived directly from Churn Probability — Low Risk (≤30%), Medium Risk (30–60%), High Risk (>60%). They encode the same underlying signal, just one continuous and one bucketed. Green = Low Risk, Red = High Risk.",
  },
  ChurnProbability: {
    label: "Coloured by Predicted Churn Probability (0–1)",
    caption: "Gradient from green (0% churn probability) to red (100%). Dense red zones are where the model is most confident about churn — your highest-priority outreach targets. Scattered red dots inside green areas are borderline customers the model is less certain about.",
  },
  UpliftScore: {
    label: "Coloured by Uplift Score (red = negative, green = positive)",
    caption: "Red = negative uplift (intervention would backfire for these customers), Yellow = neutral, Green = positive uplift (intervention helps). The scale is centred at 0. Target customers who are BOTH red-to-amber on Churn Probability AND green on Uplift Score — these are your Persuadables.",
  },
};

interface Props { summary: SegmentSummaryRow[]; umap: UmapPoint[]; customers: Customer[] }

export function SegmentationClient({ summary, umap, customers }: Props) {
  const [colorBy, setColorBy] = useState("Segment");
  const [showDefs, setShowDefs] = useState(false);

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
      const bySegment: Record<string, UmapPoint[]> = {};
      for (const p of umap) {
        if (!bySegment[p.segment]) bySegment[p.segment] = [];
        bySegment[p.segment].push(p);
      }
      return Object.entries(bySegment).map(([seg, rows]) => ({
        type: "scatter" as const,
        mode: "markers" as const,
        name: seg,
        x: rows.map((r) => r.umap_1),
        y: rows.map((r) => r.umap_2),
        text: rows.map((r) => `Customer ${r.customer_id}<br>Seg: ${r.segment}<br>Churn Prob: ${(r.churn_probability * 100).toFixed(1)}%`),
        marker: { size: 5, color: SEGMENT_COLORS[seg] ?? "#6B7280", opacity: 0.75, line: { width: 0.5, color: "white" } },
        hovertemplate: "%{text}<extra>%{fullData.name}</extra>",
      }));
    }

    const colorValues = umap.map((c) => {
      if (colorBy === "Churn") return c.churn;
      if (colorBy === "ChurnProbability") return c.churn_probability;
      if (colorBy === "UpliftScore") return c.uplift_score;
      if (colorBy === "RiskTier") return c.risk_tier === "High Risk" ? 1 : c.risk_tier === "Medium Risk" ? 0.5 : 0;
      return 0;
    });

    const colorscale = colorBy === "UpliftScore" ? COLORSCALE_HIGH_GOOD : COLORSCALE_HIGH_BAD;
    const vals = colorValues as number[];
    const absMax = Math.max(Math.abs(Math.min(...vals)), Math.abs(Math.max(...vals)));
    const cmin = colorBy === "UpliftScore" ? -absMax : undefined;
    const cmax = colorBy === "UpliftScore" ? absMax : undefined;

    return [{
      type: "scatter" as const,
      mode: "markers" as const,
      name: colorBy,
      x: umap.map((c) => c.umap_1),
      y: umap.map((c) => c.umap_2),
      marker: { size: 5, color: colorValues, colorscale, cmin, cmax, showscale: true, opacity: 0.75, line: { width: 0.5, color: "white" }, colorbar: { thickness: 14, len: 0.8, tickfont: { size: 11 } } },
      text: umap.map((c) => `Customer ${c.customer_id}<br>Seg: ${c.segment}<br>Prob: ${(c.churn_probability * 100).toFixed(1)}%`),
      hovertemplate: "%{text}<extra></extra>",
    }];
  }, [colorBy, umap]);

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
        "High ≥90%":     confs.filter((v) => v >= 0.9).length,
        "Medium 80–90%": confs.filter((v) => v >= 0.8 && v < 0.9).length,
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

      <div className="bg-[#EEF2FF] border-l-4 border-[#6366F1] rounded-r-xl px-4 py-3 mb-4 text-[14px] text-[#1E1B4B]">
        <strong>What this page shows:</strong> ~{customers.length.toLocaleString()} customers grouped into 5 behavioural segments using K-Means clustering on purchase, engagement, and satisfaction patterns. Each segment has a different churn profile and needs a different retention strategy.
      </div>

      {/* Segment definitions */}
      <button
        onClick={() => setShowDefs(!showDefs)}
        className="mb-5 px-4 py-2 rounded-xl text-[13px] font-bold border-2 border-[#DDD6FE] text-[#6366F1] bg-white hover:border-[#6366F1] transition-all"
      >
        {showDefs ? "▲ Hide" : "▼ Show"} segment definitions
      </button>
      {showDefs && (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3 mb-6">
          {Object.entries(SEGMENT_DEFINITIONS).map(([seg, def]) => (
            <div key={seg} className="rounded-2xl border-2 bg-white p-4 shadow-sm" style={{ borderColor: SEGMENT_COLORS[seg] ?? "#DDD6FE" }}>
              <div className="flex items-center gap-2 mb-1">
                <span className="w-3 h-3 rounded-full shrink-0" style={{ background: SEGMENT_COLORS[seg] }} />
                <span className="text-[14px] font-bold" style={{ color: SEGMENT_COLORS[seg] }}>{seg}</span>
              </div>
              <p className="text-[12px] font-semibold text-[#4B5563] mb-1">{def.tagline}</p>
              <p className="text-[12px] text-[#6B7280] leading-snug">{def.detail}</p>
            </div>
          ))}
        </div>
      )}

      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-4 mb-8">
        {kpiData.map((k) => (
          <MetricCard key={k.segment} label={k.segment} value={k.count.toLocaleString()} delta={`${(k.churnRate * 100).toFixed(1)}% churn rate`} accentColor={k.color} />
        ))}
      </div>

      {/* UMAP */}
      <SectionHeading>Customer Behavioural Space — UMAP 2D Projection</SectionHeading>
      <div className="flex flex-wrap items-center gap-3 mb-3">
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
            margin: { l: 30, r: 50, t: 20, b: 30 },
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
      <SectionHeading>Segment Feature Heatmap — What Makes Each Segment Different</SectionHeading>
      <div className="bg-[#F5F3FF] border border-[#DDD6FE] rounded-xl px-4 py-2.5 mb-3 text-[13px] text-[#4338CA]">
        Average value of key raw features per segment. Use this to validate the segment labels: Champions should have high tenure and cashback; Lapsed should have high days-since-last-order and low app hours; Price Sensitive should have high cashback (discount-driven). If the bars don't match the expected pattern, the segment labeling may be off.
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
      <SectionHeading>Segment Assignment Confidence (GMM Soft Probabilities)</SectionHeading>
      <div className="bg-[#F5F3FF] border border-[#DDD6FE] rounded-xl px-4 py-2.5 mb-3 text-[13px] text-[#4338CA]">
        After K-Means assigns each customer to a segment, Gaussian Mixture Models (GMM) score how <em>confident</em> that assignment is by computing a soft probability distribution across all 5 segments. <strong>Indigo = clearly belongs to one segment (≥90% confident)</strong>. Amber = sits between two segments. Red = borderline customer who warrants manual review before targeting.
      </div>
      <ChartCard>
        <ResponsiveContainer width="100%" height={420}>
          <BarChart data={gmmData} margin={{ top: 10, right: 20, left: 0, bottom: 20 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#E0E7FF" />
            <XAxis dataKey="segment" tick={{ fontSize: 12, fill: "#6B7280" }} />
            <YAxis tick={{ fontSize: 12, fill: "#6B7280" }} />
            <Tooltip contentStyle={{ borderRadius: "10px", border: "2px solid #DDD6FE", fontSize: 13 }} />
            <Legend wrapperStyle={{ fontSize: 13, paddingTop: 12 }} />
            <Bar dataKey="High ≥90%"     stackId="a" fill="#6366F1" />
            <Bar dataKey="Medium 80–90%" stackId="a" fill="#F59E0B" />
            <Bar dataKey="Boundary <80%" stackId="a" fill="#F43F5E" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </ChartCard>

      <div className="h-8" />

      {/* Summary table */}
      <SectionHeading>Segment Summary Table</SectionHeading>
      <div className="bg-[#F5F3FF] border border-[#DDD6FE] rounded-xl px-4 py-2.5 mb-3 text-[13px] text-[#4338CA]">
        Quick reference: size of each segment, observed churn rate (actual historical churners), average predicted churn probability from the XGBoost model, and the share classified as Persuadable (worth targeting with a retention campaign).
      </div>
      <div className="bg-white rounded-2xl border-2 border-[#DDD6FE] overflow-hidden shadow-sm">
        <table className="w-full text-[14px]">
          <thead>
            <tr style={{ background: "linear-gradient(110deg, #6366F1 0%, #A855F7 100%)" }}>
              {["Segment", "Customers", "Actual Churn Rate", "Avg Predicted Prob", "High Risk %", "Persuadable %"].map((h) => (
                <th key={h} className="text-white font-bold text-left px-4 py-3 text-[12px] uppercase tracking-wide">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {summary.map((s, i) => (
              <tr key={s.segment} className={i % 2 === 0 ? "bg-white" : "bg-[#F5F3FF]"}>
                <td className="px-4 py-3 font-semibold">
                  <div className="flex items-center gap-2">
                    <span className="w-3 h-3 rounded-full shrink-0" style={{ background: SEGMENT_COLORS[s.segment] ?? "#6B7280" }} />
                    {s.segment}
                  </div>
                </td>
                <td className="px-4 py-3">{s.customer_count.toLocaleString()}</td>
                <td className="px-4 py-3 font-semibold" style={{ color: s.churn_rate > 0.3 ? "#F43F5E" : s.churn_rate > 0.15 ? "#F59E0B" : "#10B981" }}>
                  {(s.churn_rate * 100).toFixed(1)}%
                </td>
                <td className="px-4 py-3">{(s.avg_churn_prob * 100).toFixed(1)}%</td>
                <td className="px-4 py-3">{(s.high_risk_pct * 100).toFixed(1)}%</td>
                <td className="px-4 py-3 font-semibold text-[#6366F1]">{(s.persuadable_pct * 100).toFixed(1)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Glossary */}
      <div className="mt-6 bg-[#F8FAFC] border border-[#E2E8F0] rounded-2xl p-5">
        <p className="text-[12px] font-bold uppercase tracking-wide text-[#64748B] mb-3">Parameter Glossary</p>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-[13px]">
          {[
            ["Actual Churn Rate", "% of customers in this segment who actually churned in the historical dataset."],
            ["Avg Predicted Prob", "Mean output of the XGBoost churn model for customers in this segment (0–100%)."],
            ["High Risk %", "% of the segment the model predicts has >60% probability of churning."],
            ["Persuadable %", "% of the segment where the T-S uplift model predicts a retention intervention would help."],
            ["UMAP", "Dimensionality reduction: 8+ behavioral features compressed to 2D for visualization while preserving cluster structure."],
            ["GMM Confidence", "Gaussian Mixture Model soft probability: how certain the model is that a customer belongs to their assigned segment."],
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
