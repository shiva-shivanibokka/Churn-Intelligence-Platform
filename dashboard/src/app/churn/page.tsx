import {
  getChurnKpis, getChurnHistogram, getRiskSummary, getShapSummary, getAvgChurnBySegment,
} from "@/lib/data";
import { ChurnClient } from "@/components/pages/churn-client";

// Cached, not force-dynamic.
//
// Every page in this dashboard was `force-dynamic`, so each sidebar click paid
// the full Supabase round-trip again — and on this page that meant waiting on a
// query before a single byte of HTML was sent. The navigation felt broken
// because it was doing real work every time, for data that had not changed.
//
// The customers table only changes when the ML pipeline is re-run and
// `restore_supabase.py` reloads it, which is a deliberate manual act, not a live
// feed. An hour is far shorter than the gap between those, so the figures are
// never meaningfully stale, and the second visitor onwards gets the page
// instantly. `/analytics` and `/retention` stay dynamic because the agent writes
// rows to them while you watch.
export const revalidate = 3600;

export default async function ChurnPage() {
  // Two waves, not three.
  //
  // This page pre-fetches every segment's data so the client-side filter is
  // instant, which is the right call — but it was doing it as a waterfall:
  // `getRiskSummary()` alone, then four "all segments" queries, then fifteen
  // per-segment ones. Twenty round-trips in three sequential waves, and the
  // slowest wave gated the two after it. Measured at 3.7 seconds before the
  // first byte of HTML.
  //
  // Only the segment *names* actually depend on the first query, so the four
  // unfiltered queries have no reason to wait for it. Running them alongside
  // leaves one genuine dependency and one batch behind it.
  const [riskSummary, kpisAll, histAll, shapAll, avgChurnBySeg] = await Promise.all([
    getRiskSummary(),
    getChurnKpis(),
    getChurnHistogram(),
    getShapSummary(),
    getAvgChurnBySegment(),
  ]);

  const segments = riskSummary.map((r) => r.segment);

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
