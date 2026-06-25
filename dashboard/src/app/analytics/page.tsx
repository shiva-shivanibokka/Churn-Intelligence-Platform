import { getRetentionActions } from "@/lib/data";
import { AnalyticsClient } from "@/components/pages/analytics-client";

export const dynamic = "force-dynamic";

export default async function AnalyticsPage() {
  const actions = await getRetentionActions(200).catch(() => []);
  return <AnalyticsClient actions={actions} />;
}
