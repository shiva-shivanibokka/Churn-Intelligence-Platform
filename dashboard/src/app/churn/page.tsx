import {
  getChurnKpis, getChurnHistogram, getRiskSummary, getShapSummary, getAvgChurnBySegment,
} from "@/lib/data";
import { ChurnClient } from "@/components/pages/churn-client";

export const dynamic = "force-dynamic";

export default async function ChurnPage() {
  const riskSummary = await getRiskSummary();
  const segments = riskSummary.map((r) => r.segment);

  const [kpisAll, histAll, shapAll, avgChurnBySeg] = await Promise.all([
    getChurnKpis(),
    getChurnHistogram(),
    getShapSummary(),
    getAvgChurnBySegment(),
  ]);

  const perSegEntries = await Promise.all(
    segments.map(async (seg) => {
      const [kpis, hist, shap] = await Promise.all([
        getChurnKpis(seg),
        getChurnHistogram(seg),
        getShapSummary(seg),
      ]);
      return [seg, { kpis, hist, shap }] as const;
    })
  );

  return (
    <ChurnClient
      kpisAll={kpisAll}
      histAll={histAll}
      shapAll={shapAll}
      riskSummary={riskSummary}
      avgChurnBySeg={avgChurnBySeg}
      segmentData={Object.fromEntries(perSegEntries)}
    />
  );
}
