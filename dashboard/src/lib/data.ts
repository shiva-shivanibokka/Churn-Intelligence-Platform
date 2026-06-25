import { supabase, Customer, RetentionAction } from "./supabase";

// ── Paginated full-customer fetch ─────────────────────────────────────────────
// Supabase caps table reads at max_rows (default 1000). We fetch all pages in
// parallel using the count from a HEAD request, then merge the results.
export async function getCustomers(): Promise<Customer[]> {
  const { count, error: countErr } = await supabase
    .from("customers")
    .select("*", { count: "exact", head: true });

  if (countErr || !count) return [];

  const PAGE_SIZE = 1000;
  const pages = Math.ceil(count / PAGE_SIZE);

  const results = await Promise.all(
    Array.from({ length: pages }, (_, i) =>
      supabase
        .from("customers")
        .select("*")
        .order("churn_probability", { ascending: false })
        .range(i * PAGE_SIZE, (i + 1) * PAGE_SIZE - 1)
    )
  );

  return results.flatMap(({ data }) => data ?? []) as Customer[];
}

// ── Individual customer lookup ────────────────────────────────────────────────
export async function getCustomer(customerId: string): Promise<Customer | null> {
  const { data, error } = await supabase
    .from("customers")
    .select("*")
    .eq("customer_id", customerId)
    .single();
  if (error) return null;
  return data;
}

// ── Segment summary RPC ───────────────────────────────────────────────────────
export type SegmentSummaryRow = {
  segment: string;
  customer_count: number;
  churn_rate: number;
  avg_churn_prob: number;
  high_risk_pct: number;
  persuadable_pct: number;
  avg_tenure: number;
  avg_satisfaction: number;
  avg_days_since_order: number;
  avg_hour_spend: number;
  avg_cashback: number;
  gmm_high: number;
  gmm_medium: number;
  gmm_boundary: number;
};

export async function getSegmentSummary(): Promise<SegmentSummaryRow[]> {
  const { data, error } = await supabase.rpc("get_segment_summary");
  if (error) throw error;
  return data ?? [];
}

// ── Churn page RPCs ───────────────────────────────────────────────────────────
export type ChurnKpis = { total: number; high_risk: number; avg_churn_prob: number; actual_churners: number };
export async function getChurnKpis(segment?: string): Promise<ChurnKpis> {
  const { data, error } = await supabase.rpc("get_churn_kpis", { p_segment: segment ?? null });
  if (error) throw error;
  return (data?.[0] ?? { total: 0, high_risk: 0, avg_churn_prob: 0, actual_churners: 0 }) as ChurnKpis;
}

export type HistogramBucket = { bucket: number; count: number };
export async function getChurnHistogram(segment?: string): Promise<HistogramBucket[]> {
  const { data, error } = await supabase.rpc("get_churn_histogram", { p_segment: segment ?? null });
  if (error) throw error;
  return data ?? [];
}

export type RiskBySegment = { segment: string; high_risk: number; medium_risk: number; low_risk: number };
export async function getRiskSummary(): Promise<RiskBySegment[]> {
  const { data, error } = await supabase.rpc("get_risk_summary");
  if (error) throw error;
  return data ?? [];
}

export type ShapFeature = { feature: string; avg_importance: number };
export async function getShapSummary(segment?: string): Promise<ShapFeature[]> {
  const { data, error } = await supabase.rpc("get_shap_summary", { p_segment: segment ?? null });
  if (error) throw error;
  return data ?? [];
}

export type AvgChurnBySeg = { segment: string; avg_churn_prob: number };
export async function getAvgChurnBySegment(): Promise<AvgChurnBySeg[]> {
  const { data, error } = await supabase.rpc("get_avg_churn_by_segment");
  if (error) throw error;
  return data ?? [];
}

// ── Lightweight scatter data (no full table scan) ────────────────────────────
export type UpliftScatterPoint = {
  customer_id: string;
  customer_type: string;
  churn_probability: number;
  uplift_score: number;
  net_roi: number;
  segment: string;
};

export async function getUpliftScatterData(): Promise<UpliftScatterPoint[]> {
  const { data, error } = await supabase
    .from("customers")
    .select("customer_id, customer_type, churn_probability, uplift_score, net_roi, segment")
    .limit(5000);
  if (error) throw error;
  return (data ?? []) as UpliftScatterPoint[];
}

export type UmapPoint = {
  customer_id: string;
  umap_1: number;
  umap_2: number;
  segment: string;
  churn_probability: number;
  churn: number;
  risk_tier: string;
  uplift_score: number;
};

export async function getUmapData(): Promise<UmapPoint[]> {
  const { data, error } = await supabase
    .from("customers")
    .select("customer_id, umap_1, umap_2, segment, churn_probability, churn, risk_tier, uplift_score")
    .limit(10000);
  if (error) throw error;
  return (data ?? []) as UmapPoint[];
}

// ── Uplift page RPCs ──────────────────────────────────────────────────────────
export type CustomerTypeSummary = {
  customer_type: string;
  count: number;
  avg_uplift_score: number;
  avg_net_roi: number;
  positive_roi_count: number;
  avg_churn_prob: number;
};
export async function getCustomerTypeSummary(): Promise<CustomerTypeSummary[]> {
  const { data, error } = await supabase.rpc("get_customer_type_summary");
  if (error) throw error;
  return data ?? [];
}

export type RoiBySegment = { segment: string; avg_roi: number; persuadable_count: number };
export async function getRoiBySegment(): Promise<RoiBySegment[]> {
  const { data, error } = await supabase.rpc("get_roi_by_segment");
  if (error) throw error;
  return data ?? [];
}

export type TopPersuadable = {
  customer_id: string;
  segment: string;
  churn_probability: number;
  uplift_score: number;
  net_roi: number;
  intervention_priority: number;
};
export async function getTopPersuadables(limit = 200): Promise<TopPersuadable[]> {
  const { data, error } = await supabase.rpc("get_top_persuadables", { p_limit: limit });
  if (error) throw error;
  return data ?? [];
}

export type UpliftKpis = {
  persuadable_count: number;
  positive_roi_count: number;
  avg_uplift_score: number;
  total_roi_potential: number;
};
export async function getUpliftKpis(): Promise<UpliftKpis> {
  const { data, error } = await supabase.rpc("get_uplift_kpis");
  if (error) throw error;
  return (data?.[0] ?? { persuadable_count: 0, positive_roi_count: 0, avg_uplift_score: 0, total_roi_potential: 0 }) as UpliftKpis;
}

// ── Retention page list ───────────────────────────────────────────────────────
export type PersuadableCustomer = {
  customer_id: string;
  segment: string;
  churn_probability: number;
  uplift_score: number;
  net_roi: number;
  customer_type: string;
};
export async function getPersuadablesList(limit = 100): Promise<PersuadableCustomer[]> {
  const { data, error } = await supabase
    .from("customers")
    .select("customer_id, segment, churn_probability, uplift_score, net_roi, customer_type")
    .eq("customer_type", "Persuadable")
    .order("net_roi", { ascending: false })
    .limit(limit);
  if (error) throw error;
  return (data ?? []) as PersuadableCustomer[];
}

// ── Retention / audit ─────────────────────────────────────────────────────────
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

export async function saveFeedback(retentionActionId: string, customerId: string, outcome: string) {
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
    total, retained, churned,
    pending: total - retained - churned,
    byType: Object.entries(byType).map(([type, v]) => ({
      type, ...v, rate: v.withFeedback ? Math.round((v.retained / v.withFeedback) * 100) : null,
    })),
    bySeg: Object.entries(bySeg).map(([segment, v]) => ({
      segment, ...v, rate: v.withFeedback ? Math.round((v.retained / v.withFeedback) * 100) : null,
    })),
  };
}
