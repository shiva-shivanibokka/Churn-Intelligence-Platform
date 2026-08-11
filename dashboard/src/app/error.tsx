"use client"; // Error boundaries must be Client Components.

import { PageTitle } from "@/components/ui/section-heading";

/**
 * Shown when a page's data load throws.
 *
 * Every page used to wrap its queries in `.catch(() => [])`, which turned "the
 * database is unreachable" into "the query returned no rows" — so the dashboard
 * rendered a confident zero-state ("~0 customers grouped into 5 segments")
 * rather than admitting it had no data. Those catches are gone; the queries now
 * throw and land here.
 *
 * This file sits beside `app/page.tsx`, so it wraps every page but *not* the
 * root layout in the same segment: the sidebar stays mounted and only the
 * content area is replaced.
 */
export default function DashboardError({
  error,
  unstable_retry,
}: {
  error: Error & { digest?: string };
  unstable_retry: () => void;
}) {
  return (
    <div className="max-w-2xl">
      <PageTitle>Dashboard unavailable</PageTitle>

      <div
        className="bg-white rounded-2xl border-2 border-[#FECACA] p-6"
        style={{ boxShadow: "0 4px 18px rgba(220,38,38,0.08)" }}
      >
        <p className="text-[15px] text-[#1E1B4B] leading-relaxed">
          The dashboard could not reach its database, so there are no figures to
          show. Nothing here is broken beyond the connection — the models, the
          scored customers and the uplift estimates are all intact, and the page
          will fill in as soon as the data source answers again.
        </p>

        <p className="mt-4 text-[14px] text-[#4B5563] leading-relaxed">
          This is the honest empty state. It is deliberately not a chart full of
          zeroes, because a zero is a measurement and this is the absence of one.
        </p>

        <div className="mt-6 flex items-center gap-4">
          <button
            onClick={() => unstable_retry()}
            className="rounded-xl px-4 py-2.5 text-[14px] font-bold text-white transition-opacity hover:opacity-90 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#4F46E5]"
            style={{
              background: "linear-gradient(110deg, #4338CA 0%, #7C3AED 100%)",
              boxShadow: "0 4px 16px rgba(67,56,202,0.28)",
            }}
          >
            Try again
          </button>

          {/* In production a Server Component's real message is withheld and only
              this hash is forwarded, so the digest is the one thing that ties
              what the visitor saw to a line in the server logs. */}
          {error.digest && (
            <span className="font-mono text-[12px] text-[#6B7280]">
              ref {error.digest}
            </span>
          )}
        </div>
      </div>
    </div>
  );
}
