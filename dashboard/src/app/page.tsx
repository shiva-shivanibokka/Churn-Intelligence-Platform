import { getSegmentSummary, getUmapData } from "@/lib/data";
import { SegmentationClient } from "@/components/pages/segmentation-client";

export const dynamic = "force-dynamic";

export default async function SegmentationPage() {
  // No .catch here, and none on any other page: a failed query throws through
  // to app/error.tsx. Catching it into [] made an unreachable database look
  // identical to an empty one, which is how this page came to announce
  // "~0 customers" as though it had counted them.
  const [summary, umap] = await Promise.all([getSegmentSummary(), getUmapData()]);
  return <SegmentationClient summary={summary} umap={umap} />;
}
