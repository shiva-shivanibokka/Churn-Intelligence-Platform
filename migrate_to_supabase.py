"""
migrate_to_supabase.py
======================
One-time script to load processed parquet data into Supabase.
Run once after pipeline.py completes.

Usage:
    python migrate_to_supabase.py

Requires DATABASE_URL in environment or .env file.
"""
import os, json, logging
from pathlib import Path

# Load .env file manually — no python-dotenv dependency needed
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROCESSED = "data/processed"

def get_conn():
    url = os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL not set. Add it to .env or environment.")
    if "sslmode" not in url:
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}sslmode=require"
    return psycopg2.connect(url, connect_timeout=15)

def create_customers_table(conn):
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS customers (
                customer_id         TEXT PRIMARY KEY,
                segment             TEXT,
                churn_probability   FLOAT,
                risk_tier           TEXT,
                uplift_score        FLOAT,
                customer_type       TEXT,
                net_roi             FLOAT,
                roi_positive        BOOLEAN,
                intervention_priority INTEGER,
                umap_1              FLOAT,
                umap_2              FLOAT,
                tenure              FLOAT,
                satisfaction_score  FLOAT,
                days_since_last_order FLOAT,
                hour_spend_on_app   FLOAT,
                complain            FLOAT,
                order_count         FLOAT,
                cashback_amount     FLOAT,
                churn               INTEGER,
                top_shap_features   JSONB,
                gmm_prob_seg0       FLOAT,
                gmm_prob_seg1       FLOAT,
                gmm_prob_seg2       FLOAT,
                gmm_prob_seg3       FLOAT,
                gmm_prob_seg4       FLOAT,
                updated_at          TIMESTAMPTZ DEFAULT NOW()
            );
            CREATE INDEX IF NOT EXISTS idx_customers_segment ON customers(segment);
            CREATE INDEX IF NOT EXISTS idx_customers_type ON customers(customer_type);
        """)
    conn.commit()
    logger.info("customers table ready.")

def load_data():
    uplift = pd.read_parquet(f"{PROCESSED}/uplift.parquet")
    logger.info("Loaded uplift.parquet: %d rows, %d cols", len(uplift), len(uplift.columns))

    col_map = {
        "CustomerID": "customer_id",
        "Segment": "segment",
        "ChurnProbability": "churn_probability",
        "RiskTier": "risk_tier",
        "UpliftScore": "uplift_score",
        "CustomerType": "customer_type",
        "NetROI": "net_roi",
        "ROIPositive": "roi_positive",
        "InterventionPriority": "intervention_priority",
        "UMAP_1": "umap_1",
        "UMAP_2": "umap_2",
        "Tenure": "tenure",
        "SatisfactionScore": "satisfaction_score",
        "DaySinceLastOrder": "days_since_last_order",
        "HourSpendOnApp": "hour_spend_on_app",
        "Complain": "complain",
        "OrderCount": "order_count",
        "CashbackAmount": "cashback_amount",
        "Churn": "churn",
        "TopSHAPFeatures": "top_shap_features",
        "GMM_Prob_Seg0": "gmm_prob_seg0",
        "GMM_Prob_Seg1": "gmm_prob_seg1",
        "GMM_Prob_Seg2": "gmm_prob_seg2",
        "GMM_Prob_Seg3": "gmm_prob_seg3",
        "GMM_Prob_Seg4": "gmm_prob_seg4",
    }

    present = {k: v for k, v in col_map.items() if k in uplift.columns}
    df = uplift[list(present.keys())].rename(columns=present)

    # Parse TopSHAPFeatures JSON string
    if "top_shap_features" in df.columns:
        def parse_shap(v):
            if isinstance(v, str):
                try: return json.loads(v)
                except: return {}
            return v if isinstance(v, dict) else {}
        df["top_shap_features"] = df["top_shap_features"].apply(parse_shap)

    # Ensure customer_id is string
    df["customer_id"] = df["customer_id"].astype(str)

    return df

def upsert_customers(conn, df):
    cols = list(df.columns)
    rows = []
    for _, row in df.iterrows():
        r = []
        for c in cols:
            v = row[c]
            if c == "top_shap_features":
                v = json.dumps(v) if v else None
            elif hasattr(v, "item"):
                v = v.item()
            elif pd.isna(v):
                v = None
            r.append(v)
        rows.append(tuple(r))

    update_cols = [c for c in cols if c != "customer_id"]
    update_sql = ", ".join(f"{c} = EXCLUDED.{c}" for c in update_cols)

    sql = f"""
        INSERT INTO customers ({', '.join(cols)})
        VALUES %s
        ON CONFLICT (customer_id) DO UPDATE SET {update_sql}, updated_at = NOW()
    """
    with conn.cursor() as cur:
        execute_values(cur, sql, rows)
    conn.commit()
    logger.info("Upserted %d customers.", len(rows))

def main():
    logger.info("Connecting to Supabase…")
    conn = get_conn()
    logger.info("Connected.")
    create_customers_table(conn)
    df = load_data()
    upsert_customers(conn, df)
    conn.close()
    logger.info("Migration complete.")

if __name__ == "__main__":
    main()
