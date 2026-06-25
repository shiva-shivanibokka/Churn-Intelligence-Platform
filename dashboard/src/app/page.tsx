import { getSegmentSummary, getUmapData, getCustomers } from "@/lib/data";
import { SegmentationClient } from "@/components/pages/segmentation-client";

export const dynamic = "force-dynamic";

export default async function SegmentationPage() {
  const [summary, umap, customers] = await Promise.all([
    getSegmentSummary(),
    getUmapData(),
    getCustomers(),
  ]);
  return <SegmentationClient summary={summary} umap={umap} customers={customers} />;
}
