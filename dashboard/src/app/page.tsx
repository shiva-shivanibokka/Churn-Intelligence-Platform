import { getSegmentSummary, getUmapData } from "@/lib/data";
import { SegmentationClient } from "@/components/pages/segmentation-client";

export const dynamic = "force-dynamic";

export default async function SegmentationPage() {
  const [summary, umap] = await Promise.all([
    getSegmentSummary().catch(() => []),
    getUmapData().catch(() => []),
  ]);
  return <SegmentationClient summary={summary} umap={umap} />;
}
