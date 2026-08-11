"""
restore_supabase.py
===================
Rebuild the entire Supabase backend from scratch, in one command.

Why this exists
---------------
The original Supabase project paused after a week of inactivity and was then
deleted at the 90-day mark, taking every table with it. Nothing of value was
lost — the data all regenerates from data/processed/uplift.parquet — but
putting it back meant running four things, from four places, in an order that
was written down nowhere:

    src/database.py::_create_schema   conversations, messages,
                                      retention_actions, intervention_feedback
    supabase/config_tables.sql        retention_playbook, business_config
    migrate_to_supabase.py            customers (schema + 51k rows)
    supabase/rpc_functions.sql        the 10 RPCs the dashboard calls
    supabase/rls_policies.sql         row-level security on all of it

Order matters: the RPCs and the RLS policies both reference `customers`, so
they have to run after the migration that creates it. This script encodes that
order so the next recovery is one command instead of an archaeology exercise.

Everything here is idempotent — every DDL statement is CREATE ... IF NOT EXISTS
or CREATE OR REPLACE, and the customer load truncates before inserting — so
running it twice is safe.

It finishes by verifying rather than assuming: it counts the rows and calls all
ten RPCs, and exits non-zero if any of them fail or come back empty. A restore
that reports success while the dashboard would still show nothing is the exact
failure mode this project already had once.

Usage:
    python restore_supabase.py

Requires DATABASE_URL in .env (Supabase → Settings → Database → Connection
string → URI). Use the pooler/session connection string if direct connection
is unavailable on your network.
"""
import logging
import sys
from pathlib import Path

# migrate_to_supabase loads .env into os.environ on import, and owns the
# customers schema + parquet load. Reuse it rather than restating either.
import migrate_to_supabase as migrate

sys.path.insert(0, str(Path(__file__).parent / "src"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SQL_DIR = Path(__file__).parent / "supabase"

# The ten functions the dashboard calls, with the arguments it calls them with.
# Kept in sync with dashboard/src/lib/data.ts.
RPCS = [
    ("get_segment_summary", ""),
    ("get_churn_kpis", "NULL"),
    ("get_churn_histogram", "NULL"),
    ("get_risk_summary", ""),
    ("get_shap_summary", "NULL"),
    ("get_avg_churn_by_segment", ""),
    ("get_customer_type_summary", ""),
    ("get_roi_by_segment", ""),
    ("get_top_persuadables", "10"),
    ("get_uplift_kpis", ""),
]


def run_sql_file(conn, name: str) -> None:
    """Execute a .sql file as a single batch.

    psycopg2 only treats % as a placeholder when parameters are passed, so the
    dollar-quoted function bodies in rpc_functions.sql go through untouched.
    """
    path = SQL_DIR / name
    with conn.cursor() as cur:
        cur.execute(path.read_text(encoding="utf-8"))
    conn.commit()
    logger.info("Applied %s", name)


def create_agent_tables(conn) -> None:
    """conversations, messages, retention_actions, intervention_feedback.

    These live in src/database.py because the agent creates them at runtime;
    calling that same function keeps one definition rather than a second copy
    that can drift.
    """
    import database

    database._create_schema(conn)
    conn.commit()
    logger.info("Applied src/database.py schema (agent + feedback tables)")


def verify(conn) -> bool:
    """Prove the dashboard would actually render, and say so honestly."""
    ok = True

    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM customers")
        n_customers = cur.fetchone()[0]
        cur.execute("SELECT COUNT(DISTINCT segment) FROM customers")
        n_segments = cur.fetchone()[0]

    logger.info("customers: %s rows across %s segments", f"{n_customers:,}", n_segments)
    if n_customers == 0:
        logger.error("customers table is empty — the dashboard would show nothing.")
        ok = False

    for fn, args in RPCS:
        try:
            with conn.cursor() as cur:
                cur.execute(f"SELECT * FROM {fn}({args})")
                rows = cur.fetchall()
        except Exception as exc:
            logger.error("RPC %s FAILED: %s", fn, exc)
            conn.rollback()
            ok = False
            continue
        if not rows:
            logger.error("RPC %s returned no rows", fn)
            ok = False
        else:
            logger.info("RPC %-26s %d row(s)", fn, len(rows))

    for table in ("retention_playbook", "business_config"):
        with conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            count = cur.fetchone()[0]
        logger.info("%-20s %d row(s)", table, count)
        if count == 0:
            logger.error("%s is empty — the AI agent reads its config from here.", table)
            ok = False

    return ok


def main() -> int:
    logger.info("Connecting…")
    conn = migrate.get_conn()
    logger.info("Connected.")

    # 1. Tables with no dependencies.
    create_agent_tables(conn)
    run_sql_file(conn, "config_tables.sql")

    # 2. customers — must exist before the RPCs and policies that reference it.
    migrate.create_customers_table(conn)
    migrate.truncate_customers(conn)
    df = migrate.load_data()
    migrate.upsert_customers(conn, df)

    # 3. Everything that depends on customers.
    run_sql_file(conn, "rpc_functions.sql")
    run_sql_file(conn, "rls_policies.sql")

    logger.info("--- verifying ---")
    ok = verify(conn)
    conn.close()

    if ok:
        logger.info("Restore complete and verified.")
        return 0
    logger.error("Restore finished with problems — see the errors above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
