import type { NextRequest } from "next/server";
import { supabase } from "@/lib/supabase";

/**
 * Keeps the Supabase project from being paused for inactivity.
 *
 * This endpoint exists because of a real failure, not a hypothetical one. The
 * original Supabase project paused after a week without traffic, nobody
 * noticed, and free-tier paused projects are deleted after 90 days — so the
 * dashboard's entire backend was reclaimed while the site sat there serving an
 * empty UI. Recreating the project without this would just restart that clock.
 *
 * Supabase counts any API request as activity, so one cheap SELECT a day keeps
 * the project permanently inside its 7-day window and the deletion timer never
 * starts. `business_config` has four rows and is the smallest real table here.
 *
 * The query is deliberately allowed to fail loudly: returning 500 makes a
 * broken keepalive show up as a failed cron in Vercel's logs. A keepalive that
 * silently returns 200 when the database is gone is worse than none at all,
 * because it reports success at exactly the moment it has stopped working.
 */
export const dynamic = "force-dynamic";

export async function GET(request: NextRequest) {
  // Fails closed: with CRON_SECRET unset, nothing is authorised. Vercel sends
  // this header on scheduled invocations once the variable is set on the
  // project.
  const cronSecret = process.env.CRON_SECRET;
  const authHeader = request.headers.get("authorization");
  if (!cronSecret || authHeader !== `Bearer ${cronSecret}`) {
    return new Response("Unauthorized", { status: 401 });
  }

  const startedAt = Date.now();
  const { data, error } = await supabase
    .from("business_config")
    .select("key")
    .limit(1);

  if (error) {
    console.error("keepalive: Supabase unreachable —", error.message);
    return Response.json(
      { ok: false, error: error.message, ms: Date.now() - startedAt },
      { status: 500 }
    );
  }

  return Response.json({
    ok: true,
    rows: data?.length ?? 0,
    ms: Date.now() - startedAt,
  });
}
