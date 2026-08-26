import {
  getUpliftKpis,
  getCustomerTypeSummary,
  getRoiBySegment,
  getTopPersuadables,
  getUpliftScatterData,
} from "@/lib/data";
import { UpliftClient } from "@/components/pages/uplift-client";

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

export default async function UpliftPage() {
  const [kpis, typeSummary, roiBySeg, topPersuadables, scatter] = await Promise.all([
    getUpliftKpis(),
    getCustomerTypeSummary(),
    getRoiBySegment(),
    // 250, not 2,000. This feeds a "campaign priority list" ordered by ROI,
    // and a table nobody scrolls past the first screen of was costing about a
    // megabyte to render — the rows below the top few hundred are not a
    // priority list, they are the rest of the segment.
    getTopPersuadables(250),
    getUpliftScatterData(),
  ]);
  return (
    <UpliftClient
      kpis={kpis}
      typeSummary={typeSummary}
      roiBySeg={roiBySeg}
      topPersuadables={topPersuadables}
      scatter={scatter}
    />
  );
}
