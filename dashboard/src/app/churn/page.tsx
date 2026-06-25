import {
  getChurnKpis, getChurnHistogram, getRiskSummary, getShapSummary, getAvgChurnBySegment,
  ChurnKpis, HistogramBucket, ShapFeature,
} from "@/lib/data";
import { ChurnClient } from "@/components/pages/churn-client";

export const dynamic = "force-dynamic";

const EMPTY_KPIS: ChurnKpis = { total: 0, high_risk: 0, avg_churn_prob: 0, actual_churners: 0 };

export default async function ChurnPage() {
  const riskSummary = await getRiskSummary().catch(() => []);
  const segments = riskSummary.map((r) => r.segment);

  const [kpisAll, histAll, shapAll, avgChurnBySeg] = await Promise.all([
    getChurnKpis().catch(() => EMPTY_KPIS),
    getChurnHistogram().catch(() => [] as HistogramBucket[]),
    getShapSummary().catch(() => [] as ShapFeature[]),
    getAvgChurnBySegment().catch(() => []),
  ]);

  const perSegEntries = await Promise.all(
    segments.map(async (seg) => {
      const [kpis, hist, shap] = await Promise.all([
        getChurnKpis(seg).catch(() => EMPTY_KPIS),
        getChurnHistogram(seg).catch(() => [] as HistogramBucket[]),
        getShapSummary(seg).catch(() => [] as ShapFeature[]),
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
