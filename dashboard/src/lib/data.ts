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

// ── Payload trimming ──────────────────────────────────────────────────────────

/**
 * Round a float to `dp` decimals for transport.
 *
 * The scatter plots send thousands of rows, and a float64 serialises to its
 * full precision: `-0.05342163741588593` is twenty characters to express a
 * number the chart draws at four. Across 10,000 points and several columns that
 * was the difference between a 2.3 MB page and a manageable one — and every one
 * of those bytes is paid for twice, once in the server-rendered HTML and again
 * in the RSC flight payload beside it.
 *
 * Four decimals is well past what a scatter pixel or a percentage label can
 * show. Nothing displayed changes.
 */
function trim(value: number | null | undefined, dp = 4): number {
  if (value == null || Number.isNaN(value)) return 0;
  const factor = 10 ** dp;
  return Math.round(value * factor) / factor;
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
  return (data ?? []).map((c) => ({
    customer_id: c.customer_id,
    customer_type: c.customer_type,
    segment: c.segment,
    churn_probability: trim(c.churn_probability),
    uplift_score: trim(c.uplift_score),
    net_roi: trim(c.net_roi, 2),
  })) as UpliftScatterPoint[];
}

/**
 * The behavioural-space scatter, as columns.
 *
 * Sent column-wise rather than as an array of row objects, because the row
 * shape spends most of its bytes saying the same thing ten thousand times.
 * A single row serialised to about 170 bytes, of which ~95 were the JSON key
 * names — `customer_id`, `churn_probability`, `uplift_score` and the rest,
 * repeated once per customer. That is roughly a megabyte of keys in a 2.3 MB
 * page, and every byte is parsed on the main thread before React can hydrate,
 * which is why the page rendered long before it would respond to a click.
 *
 * Columns say each name once. Segments are additionally factorised into a
 * codebook, so "Price Sensitive" is stored as a small integer rather than
 * fourteen characters per point.
 *
 * `risk_tier` is gone: it was only ever used to colour points by tier, and the
 * tier is a fixed function of the churn probability (0.3 / 0.6), so it is
 * derived on the client instead of shipped.
 */
export type UmapColumns = {
  /** Segment names, indexed by `seg`. */
  segments: string[];
  ids: string[];
  seg: number[];
  x: number[];
  y: number[];
  prob: number[];
  churn: number[];
  uplift: number[];
};

export async function getUmapData(): Promise<UmapColumns> {
  const { data, error } = await supabase
    .from("customers")
    .select("customer_id, umap_1, umap_2, segment, churn_probability, churn, uplift_score")
    .limit(10000);
  if (error) throw error;

  const rows = data ?? [];
  const segments: string[] = [];
  const segIndex = new Map<string, number>();

  const columns: UmapColumns = {
    segments,
    ids: [], seg: [], x: [], y: [], prob: [], churn: [], uplift: [],
  };

  for (const r of rows) {
    let idx = segIndex.get(r.segment);
    if (idx === undefined) {
      idx = segments.push(r.segment) - 1;
      segIndex.set(r.segment, idx);
    }
    columns.ids.push(r.customer_id);
    columns.seg.push(idx);
    // Two decimals on the coordinates: the plot is ~700px across a range of
    // about 30 units, so anything finer lands on the same pixel.
    columns.x.push(trim(r.umap_1, 2));
    columns.y.push(trim(r.umap_2, 2));
    columns.prob.push(trim(r.churn_probability));
    columns.churn.push(r.churn);
    columns.uplift.push(trim(r.uplift_score));
  }

  return columns;
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
  return ((data ?? []) as TopPersuadable[]).map((c) => ({
    ...c,
    churn_probability: trim(c.churn_probability),
    uplift_score: trim(c.uplift_score),
    net_roi: trim(c.net_roi, 2),
  }));
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

