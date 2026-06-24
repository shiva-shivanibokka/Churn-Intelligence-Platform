import { supabase, Customer, RetentionAction } from "./supabase";

export async function getCustomers(): Promise<Customer[]> {
  const { data, error } = await supabase
    .from("customers")
    .select("*")
    .order("churn_probability", { ascending: false });
  if (error) throw error;
  return data ?? [];
}

export async function getCustomer(customerId: string): Promise<Customer | null> {
  const { data, error } = await supabase
    .from("customers")
    .select("*")
    .eq("customer_id", customerId)
    .single();
  if (error) return null;
  return data;
}

export async function getRetentionActions(limit = 200): Promise<RetentionAction[]> {
  const [{ data: actions, error }, { data: feedback }] = await Promise.all([
    supabase.from("retention_actions").select("*").order("generated_at", { ascending: false }).limit(limit),
    supabase.from("intervention_feedback").select("retention_action_id, outcome"),
  ]);
  if (error) throw error;
  const fbMap: Record<string, string> = {};
  for (const f of feedback ?? []) fbMap[f.retention_action_id] = f.outcome;
  return (actions ?? []).map((a) => ({ ...a, outcome: fbMap[a.id] ?? null })) as RetentionAction[];
}

export async function saveFeedback(
  retentionActionId: string,
  customerId: string,
  outcome: string
) {
  const { error } = await supabase.from("intervention_feedback").insert({
    id: crypto.randomUUID(),
    retention_action_id: retentionActionId,
    customer_id: customerId,
    outcome,
  });
  if (error) throw error;
}

export async function getAuditSummary() {
  const [{ data: actions }, { data: feedback }] = await Promise.all([
    supabase.from("retention_actions").select("id, intervention_type, segment, generated_at"),
    supabase.from("intervention_feedback").select("retention_action_id, outcome"),
  ]);

  const total = actions?.length ?? 0;
  const feedbackMap = Object.fromEntries(
    (feedback ?? []).map((f) => [f.retention_action_id, f.outcome])
  );

  const retained = Object.values(feedbackMap).filter((o) => o === "retained").length;
  const churned = Object.values(feedbackMap).filter((o) => o === "churned").length;

  // By intervention type
  const byType: Record<string, { total: number; retained: number; withFeedback: number }> = {};
  for (const a of actions ?? []) {
    const t = a.intervention_type ?? "Unknown";
    if (!byType[t]) byType[t] = { total: 0, retained: 0, withFeedback: 0 };
    byType[t].total++;
    if (feedbackMap[a.id]) {
      byType[t].withFeedback++;
      if (feedbackMap[a.id] === "retained") byType[t].retained++;
    }
  }

  // By segment
  const bySeg: Record<string, { total: number; retained: number; withFeedback: number }> = {};
  for (const a of actions ?? []) {
    const s = a.segment ?? "Unknown";
    if (!bySeg[s]) bySeg[s] = { total: 0, retained: 0, withFeedback: 0 };
    bySeg[s].total++;
    if (feedbackMap[a.id]) {
      bySeg[s].withFeedback++;
      if (feedbackMap[a.id] === "retained") bySeg[s].retained++;
    }
  }

  return {
    total,
    retained,
    churned,
    pending: total - retained - churned,
    byType: Object.entries(byType).map(([type, v]) => ({
      type,
      ...v,
      rate: v.withFeedback ? Math.round((v.retained / v.withFeedback) * 100) : null,
    })),
    bySeg: Object.entries(bySeg).map(([segment, v]) => ({
      segment,
      ...v,
      rate: v.withFeedback ? Math.round((v.retained / v.withFeedback) * 100) : null,
    })),
  };
}
