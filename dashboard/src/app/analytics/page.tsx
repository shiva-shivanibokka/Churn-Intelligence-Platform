import { getRetentionActions, getAuditSummary } from "@/lib/data";
import { AnalyticsClient } from "@/components/pages/analytics-client";

export const revalidate = 0;

const EMPTY_SUMMARY = { total: 0, retained: 0, churned: 0, pending: 0, byType: [], bySeg: [] };

export default async function AnalyticsPage() {
  const [actions, summary] = await Promise.all([
    getRetentionActions(200).catch(() => []),
    getAuditSummary().catch(() => EMPTY_SUMMARY),
  ]);
  return <AnalyticsClient actions={actions} summary={summary} />;
}
