import { getSegmentSummary, getUmapData } from "@/lib/data";
import { SegmentationClient } from "@/components/pages/segmentation-client";

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

export default async function SegmentationPage() {
  // No .catch here, and none on any other page: a failed query throws through
  // to app/error.tsx. Catching it into [] made an unreachable database look
  // identical to an empty one, which is how this page came to announce
  // "~0 customers" as though it had counted them.
  const [summary, umap] = await Promise.all([getSegmentSummary(), getUmapData()]);
  return <SegmentationClient summary={summary} umap={umap} />;
}
