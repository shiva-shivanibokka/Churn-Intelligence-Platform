import { getPersuadablesList } from "@/lib/data";
import { RetentionClient } from "@/components/pages/retention-client";

export const dynamic = "force-dynamic";

export default async function RetentionPage() {
  const persuadables = await getPersuadablesList(100).catch(() => []);
  return <RetentionClient persuadables={persuadables} />;
}
