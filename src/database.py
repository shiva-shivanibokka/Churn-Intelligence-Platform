"""
Database Layer — PostgreSQL via psycopg2
=========================================
Provides persistence for:
  - Conversation history (multi-turn chat, survives browser refresh)
  - Retention action audit trail (compliance log of every LLM recommendation)
  - Intervention feedback (CSM marks outcome: retained / churned / pending)

Graceful degradation: if DATABASE_URL is not set or connection fails, all
functions return None/empty and the app continues working with session_state
only. This allows the app to run locally without a database configured.

Production equivalent: AWS RDS PostgreSQL / Cloud SQL / Supabase.
Connection pooling equivalent: PgBouncer / Supabase Pooler.
"""

import json
import logging
import os
import uuid
from contextlib import contextmanager
from typing import Optional

logger = logging.getLogger(__name__)

_DATABASE_URL: Optional[str] = None
_db_available: bool = False

# ─── Initialisation ───────────────────────────────────────────────────────────

def initialize(database_url: Optional[str] = None) -> bool:
    """
    Connect to PostgreSQL and create schema if tables don't exist.
    Call once at app startup. Returns True if connection succeeded.
    """
    global _DATABASE_URL, _db_available

    url = database_url or os.getenv("DATABASE_URL")
    if not url:
        logger.info("DATABASE_URL not configured — running without persistence.")
        return False

    # Embed sslmode=require in the URL — more portable than a keyword arg
    if "sslmode" not in url:
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}sslmode=require"
    _DATABASE_URL = url

    try:
        import psycopg2  # imported here so missing package doesn't crash the app
        with _get_conn() as conn:
            _create_schema(conn)
        _db_available = True
        logger.info("Database connected and schema initialised successfully.")
        return True
    except ImportError:
        logger.warning("psycopg2 not installed — running without persistence.")
        _db_available = False
        return False
    except Exception as e:
        logger.warning(
            "Database unavailable (%s: %s) — running without persistence.",
            type(e).__name__, e,
        )
        # Try once more without SSL in case this is a local dev environment
        try:
            _DATABASE_URL = url.replace("&sslmode=require", "").replace("?sslmode=require", "")
            with _get_conn() as conn:
                _create_schema(conn)
            _db_available = True
            logger.info("Database connected (no SSL) and schema initialised.")
            return True
        except Exception:
            _db_available = False
            return False


def is_available() -> bool:
    return _db_available


@contextmanager
def _get_conn():
    import psycopg2
    conn = psycopg2.connect(_DATABASE_URL, connect_timeout=10)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _create_schema(conn):
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id          TEXT PRIMARY KEY,
                session_id  TEXT NOT NULL,
                created_at  TIMESTAMPTZ DEFAULT NOW()
            );

            CREATE TABLE IF NOT EXISTS messages (
                id               TEXT PRIMARY KEY,
                conversation_id  TEXT REFERENCES conversations(id) ON DELETE CASCADE,
                role             TEXT NOT NULL,
                content          TEXT NOT NULL,
                tool_calls       JSONB,
                created_at       TIMESTAMPTZ DEFAULT NOW()
            );

            CREATE TABLE IF NOT EXISTS retention_actions (
                id                TEXT PRIMARY KEY,
                customer_id       TEXT NOT NULL,
                segment           TEXT,
                churn_probability FLOAT,
                uplift_score      FLOAT,
                net_roi           FLOAT,
                intervention_type TEXT,
                channel           TEXT,
                timing            TEXT,
                message_framing   TEXT,
                agent_reasoning   JSONB,
                agentic_mode      BOOLEAN DEFAULT FALSE,
                generated_at      TIMESTAMPTZ DEFAULT NOW()
            );

            CREATE TABLE IF NOT EXISTS intervention_feedback (
                id                   TEXT PRIMARY KEY,
                retention_action_id  TEXT NOT NULL,
                customer_id          TEXT NOT NULL,
                outcome              TEXT,
                logged_at            TIMESTAMPTZ DEFAULT NOW()
            );

            CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id);
            CREATE INDEX IF NOT EXISTS idx_actions_customer ON retention_actions(customer_id);
            CREATE INDEX IF NOT EXISTS idx_actions_generated ON retention_actions(generated_at DESC);
            CREATE INDEX IF NOT EXISTS idx_feedback_action ON intervention_feedback(retention_action_id);
        """)


# ─── Conversation Management ──────────────────────────────────────────────────

def create_conversation(session_id: str) -> Optional[str]:
    if not _db_available:
        return None
    conv_id = str(uuid.uuid4())
    try:
        with _get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO conversations (id, session_id) VALUES (%s, %s)",
                    (conv_id, session_id),
                )
        return conv_id
    except Exception as e:
        logger.warning("create_conversation failed: %s", e)
        return None


def save_message(
    conversation_id: str,
    role: str,
    content: str,
    tool_calls: Optional[list] = None,
) -> Optional[str]:
    if not _db_available or not conversation_id:
        return None
    msg_id = str(uuid.uuid4())
    try:
        with _get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO messages (id, conversation_id, role, content, tool_calls) VALUES (%s, %s, %s, %s, %s)",
                    (msg_id, conversation_id, role, content, json.dumps(tool_calls) if tool_calls else None),
                )
        return msg_id
    except Exception as e:
        logger.warning("save_message failed: %s", e)
        return None


def load_conversation_messages(session_id: str) -> list:
    """Load all messages for the most recent conversation in this session."""
    if not _db_available:
        return []
    try:
        with _get_conn() as conn:
            with conn.cursor() as cur:
                # Get most recent conversation for this session
                cur.execute(
                    "SELECT id FROM conversations WHERE session_id = %s ORDER BY created_at DESC LIMIT 1",
                    (session_id,),
                )
                row = cur.fetchone()
                if not row:
                    return []
                conv_id = row[0]

                cur.execute(
                    "SELECT role, content, tool_calls FROM messages WHERE conversation_id = %s ORDER BY created_at",
                    (conv_id,),
                )
                rows = cur.fetchall()
        return [
            {
                "role": r[0],
                "content": r[1],
                "trace": json.loads(r[2]) if r[2] else [],
            }
            for r in rows
            if r[0] in ("user", "assistant")  # only display roles
        ]
    except Exception as e:
        logger.warning("load_conversation_messages failed: %s", e)
        return []


# ─── Retention Action Audit Trail ─────────────────────────────────────────────

def save_retention_action(action: dict, agentic_mode: bool = False) -> Optional[str]:
    """Log every generated retention action to the audit trail."""
    if not _db_available:
        return None
    action_id = str(uuid.uuid4())
    try:
        with _get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO retention_actions
                       (id, customer_id, segment, churn_probability, uplift_score, net_roi,
                        intervention_type, channel, timing, message_framing, agent_reasoning, agentic_mode)
                       VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                    (
                        action_id,
                        str(action.get("customer_id", "")),
                        action.get("segment"),
                        action.get("churn_probability"),
                        action.get("uplift_score"),
                        action.get("net_roi"),
                        action.get("intervention_type"),
                        action.get("channel"),
                        action.get("timing"),
                        action.get("message_framing"),
                        json.dumps(action.get("trace", [])),
                        agentic_mode,
                    ),
                )
        return action_id
    except Exception as e:
        logger.warning("save_retention_action failed: %s", e)
        return None


def save_feedback(retention_action_id: str, customer_id: str, outcome: str) -> Optional[str]:
    if not _db_available:
        return None
    fb_id = str(uuid.uuid4())
    try:
        with _get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO intervention_feedback (id, retention_action_id, customer_id, outcome) VALUES (%s,%s,%s,%s)",
                    (fb_id, retention_action_id, customer_id, outcome),
                )
        return fb_id
    except Exception as e:
        logger.warning("save_feedback failed: %s", e)
        return None


# ─── Analytics Queries ────────────────────────────────────────────────────────

def get_audit_summary() -> dict:
    if not _db_available:
        return {}
    try:
        with _get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM retention_actions")
                total = cur.fetchone()[0]

                cur.execute(
                    "SELECT outcome, COUNT(*) FROM intervention_feedback GROUP BY outcome"
                )
                outcomes = {row[0]: row[1] for row in cur.fetchall()}

                cur.execute(
                    """SELECT intervention_type, COUNT(*) as cnt,
                       COUNT(f.id) as with_feedback,
                       SUM(CASE WHEN f.outcome='retained' THEN 1 ELSE 0 END) as retained
                       FROM retention_actions ra
                       LEFT JOIN intervention_feedback f ON f.retention_action_id = ra.id
                       GROUP BY intervention_type ORDER BY cnt DESC"""
                )
                by_type = [
                    {
                        "intervention_type": r[0],
                        "total": r[1],
                        "with_feedback": r[2],
                        "retained": r[3],
                        "retention_rate": round(r[3] / r[2], 3) if r[2] else None,
                    }
                    for r in cur.fetchall()
                ]

                cur.execute(
                    """SELECT segment, COUNT(*) as cnt,
                       SUM(CASE WHEN f.outcome='retained' THEN 1 ELSE 0 END) as retained,
                       COUNT(f.id) as with_feedback
                       FROM retention_actions ra
                       LEFT JOIN intervention_feedback f ON f.retention_action_id = ra.id
                       WHERE segment IS NOT NULL
                       GROUP BY segment ORDER BY cnt DESC"""
                )
                by_segment = [
                    {
                        "segment": r[0],
                        "total": r[1],
                        "retained": r[2],
                        "with_feedback": r[3],
                        "retention_rate": round(r[2] / r[3], 3) if r[3] else None,
                    }
                    for r in cur.fetchall()
                ]

        return {
            "total_actions": total,
            "outcomes": outcomes,
            "by_intervention_type": by_type,
            "by_segment": by_segment,
        }
    except Exception as e:
        logger.warning("get_audit_summary failed: %s", e)
        return {}


def get_all_retention_actions(limit: int = 200) -> list:
    if not _db_available:
        return []
    try:
        with _get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT ra.id, ra.customer_id, ra.segment, ra.churn_probability,
                              ra.intervention_type, ra.channel, ra.timing,
                              ra.agentic_mode, ra.generated_at,
                              f.outcome
                       FROM retention_actions ra
                       LEFT JOIN intervention_feedback f ON f.retention_action_id = ra.id
                       ORDER BY ra.generated_at DESC
                       LIMIT %s""",
                    (limit,),
                )
                rows = cur.fetchall()
        return [
            {
                "id": r[0],
                "customer_id": r[1],
                "segment": r[2],
                "churn_probability": r[3],
                "intervention_type": r[4],
                "channel": r[5],
                "timing": r[6],
                "agentic_mode": r[7],
                "generated_at": str(r[8]),
                "outcome": r[9] or "pending",
            }
            for r in rows
        ]
    except Exception as e:
        logger.warning("get_all_retention_actions failed: %s", e)
        return []
