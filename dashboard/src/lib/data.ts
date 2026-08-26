import { supabase, RetentionAction } from "./supabase";

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
  const { data: actions, error } = await supabase
    .from("retention_actions")
    .select("*")
    .order("generated_at", { ascending: false })
    .limit(limit);
  if (error) throw error;
  if (!actions?.length) return [];

  // Fetch feedback only for the actions being shown, rather than the whole
  // table. Two reasons: an unfiltered select grows without bound as feedback
  // accumulates and eventually meets PostgREST's row cap, at which point the
  // map silently loses entries and outcomes start rendering as "pending"; and
  // this page only ever needs the rows it is about to join against.
  const ids = actions.map((a) => a.id as string);
  const { data: feedback, error: feedbackError } = await supabase
    .from("intervention_feedback")
    .select("retention_action_id, outcome")
    .in("retention_action_id", ids);

  // This error used to be destructured away. An unreachable feedback table then
  // looked exactly like "nobody has recorded an outcome yet" — every action
  // rendering as pending, no warning anywhere. That is the same collapse of
  // "unreachable" into "empty" that made the whole dashboard report zeroes
  // during the Supabase pause, so it throws to error.tsx like everything else.
  if (feedbackError) throw feedbackError;

  const fbMap: Record<string, string> = {};
  for (const f of feedback ?? []) fbMap[f.retention_action_id] = f.outcome;
  return actions.map((a) => ({ ...a, outcome: fbMap[a.id] ?? null })) as RetentionAction[];
}

export async function saveFeedback(retentionActionId: string, customerId: string, outcome: string) {
  // Upsert, not insert. A CSM who marks an action "retained" and then corrects
  // it to "churned" was leaving two rows behind, and the map in
  // getRetentionActions keys on retention_action_id — so which correction won
  // depended on the order Postgres returned the rows in. The outcome of an
  // intervention is one fact, so it gets one row.
  const { error } = await supabase.from("intervention_feedback").upsert(
    {
      id: crypto.randomUUID(),
      retention_action_id: retentionActionId,
      customer_id: customerId,
      outcome,
    },
    { onConflict: "retention_action_id" }
  );
  if (error) throw error;
}

