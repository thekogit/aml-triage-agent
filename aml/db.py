"""Postgres + pgvector serving store. Two classes, two roles:
PgRepo  -> Repo protocol, connects as SELECT-only role `aml_agent` (cannot see alert_truth or pesel).
PgStore -> ReviewStore + EvalStore, connects as owner `aml`. The only writers in the online system."""
from __future__ import annotations

import json
from datetime import datetime
from typing import Optional
from urllib.parse import urlsplit

import pandas as pd

from . import config
from .schema import TX_COLUMNS, AlertStatus, Decision, PolicyChunk, TriageRun

AGENT_ACCOUNT_COLUMNS = ["account", "customer_type", "name", "iban", "risk_rating",
                         "declared_monthly_turnover_usd", "onboarding_date", "country"]  # no pesel


def _agent_password() -> str:
    parsed = urlsplit(config.PG_AGENT_DSN)
    return parsed.password or "aml_agent"


_pwd = _agent_password().replace("'", "''")

SCHEMA_SQL = f"""
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS transactions (
  tx_id BIGINT PRIMARY KEY, ts TIMESTAMP NOT NULL, src TEXT NOT NULL, dst TEXT NOT NULL,
  amount_paid DOUBLE PRECISION, payment_currency TEXT, amount_received DOUBLE PRECISION,
  receiving_currency TEXT, payment_format TEXT, amount_usd DOUBLE PRECISION, memo TEXT);
CREATE INDEX IF NOT EXISTS tx_src_ts ON transactions (src, ts);
CREATE INDEX IF NOT EXISTS tx_dst_ts ON transactions (dst, ts);
CREATE TABLE IF NOT EXISTS accounts (
  account TEXT PRIMARY KEY, customer_type TEXT, name TEXT, pesel TEXT, iban TEXT, risk_rating TEXT,
  declared_monthly_turnover_usd DOUBLE PRECISION, onboarding_date DATE, country TEXT);
CREATE TABLE IF NOT EXISTS alerts (
  alert_id BIGINT PRIMARY KEY, tx_id BIGINT NOT NULL REFERENCES transactions, score DOUBLE PRECISION,
  score_rank INT, score_rank_pct DOUBLE PRECISION, reasons JSONB,
  status TEXT NOT NULL DEFAULT 'OPEN' CHECK (status IN ('OPEN','TRIAGED','DECIDED')));
CREATE TABLE IF NOT EXISTS alert_truth (
  alert_id BIGINT PRIMARY KEY REFERENCES alerts, is_laundering INT NOT NULL, pattern TEXT);
CREATE TABLE IF NOT EXISTS policy_chunks (
  chunk_id SERIAL PRIMARY KEY, doc_id TEXT, title TEXT, text TEXT, embedding vector({config.EMBED_DIM}));
CREATE TABLE IF NOT EXISTS assessments (
  assessment_id SERIAL PRIMARY KEY, alert_id BIGINT NOT NULL REFERENCES alerts, created_at TIMESTAMP DEFAULT now(),
  model TEXT, final JSONB, llm_raw JSONB, guard_reasons JSONB, grounding JSONB, tool_log JSONB,
  latency_s DOUBLE PRECISION, cost_usd DOUBLE PRECISION);
CREATE TABLE IF NOT EXISTS decisions (
  decision_id SERIAL PRIMARY KEY, alert_id BIGINT NOT NULL REFERENCES alerts,
  assessment_id INT REFERENCES assessments, analyst TEXT NOT NULL CHECK (length(analyst) >= 2),
  action TEXT NOT NULL CHECK (action IN ('APPROVE','OVERRIDE_ESCALATE','OVERRIDE_CLOSE')),
  comment TEXT, decided_at TIMESTAMP DEFAULT now(),
  CHECK (action = 'APPROVE' OR length(trim(coalesce(comment, ''))) > 0));

-- audit trail is append-only (triggers raise errors on UPDATE/DELETE to prevent silent no-ops)
CREATE OR REPLACE FUNCTION prevent_audit_modification() RETURNS TRIGGER AS $$
BEGIN
  RAISE EXCEPTION 'Table % is append-only: UPDATE and DELETE are prohibited', TG_TABLE_NAME;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS assessments_no_update_delete ON assessments;
CREATE TRIGGER assessments_no_update_delete
BEFORE UPDATE OR DELETE ON assessments
FOR EACH ROW EXECUTE FUNCTION prevent_audit_modification();

DROP TRIGGER IF EXISTS decisions_no_update_delete ON decisions;
CREATE TRIGGER decisions_no_update_delete
BEFORE UPDATE OR DELETE ON decisions
FOR EACH ROW EXECUTE FUNCTION prevent_audit_modification();

-- least-privilege role for the agent: no alert_truth, no pesel, no writes
DO $$ BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'aml_agent') THEN
    CREATE ROLE aml_agent LOGIN PASSWORD '{_pwd}';
  ELSE
    ALTER ROLE aml_agent WITH PASSWORD '{_pwd}';
  END IF;
END $$;
REVOKE ALL ON ALL TABLES IN SCHEMA public FROM aml_agent;
GRANT SELECT ON transactions, alerts, policy_chunks TO aml_agent;
GRANT SELECT ({", ".join(AGENT_ACCOUNT_COLUMNS)}) ON accounts TO aml_agent;
"""


def connect(dsn: str = config.PG_DSN):
    """psycopg.connect(dsn, autocommit=True, row_factory=dict_row) + pgvector register_vector(conn)."""
    import psycopg
    from pgvector.psycopg import register_vector
    from psycopg.rows import dict_row

    conn = psycopg.connect(dsn, autocommit=True, row_factory=dict_row)
    try:
        register_vector(conn)
    except Exception:
        pass
    return conn


def init_schema(conn) -> None:
    """conn.execute(SCHEMA_SQL). Owner connection only. Idempotent."""
    conn.execute(SCHEMA_SQL)


def copy_frame(conn, table: str, df: pd.DataFrame, columns: list[str]) -> int:
    """Bulk insert with COPY; NaN/NaT -> None; returns row count."""
    clean = df[columns].astype(object).where(df[columns].notna(), None)
    with conn.cursor() as cur, cur.copy(f"COPY {table} ({', '.join(columns)}) FROM STDIN") as cp:
        for row in clean.itertuples(index=False, name=None):
            py_row = tuple(v.item() if hasattr(v, "item") and callable(v.item) else v for v in row)
            cp.write_row(py_row)
    return len(clean)


class PgRepo:
    """Read-only Repo over Postgres. MUST be constructed with connect(config.PG_AGENT_DSN)."""

    def __init__(self, conn, embedder=None):
        self.conn, self._embedder = conn, embedder

    def alert(self, alert_id: int) -> Optional[dict]:
        return self.conn.execute(
            "SELECT a.alert_id, a.score, a.score_rank, a.score_rank_pct, a.reasons::text AS reasons, t.* "
            "FROM alerts a JOIN transactions t USING (tx_id) WHERE a.alert_id = %s",
            (alert_id,),
        ).fetchone()

    def profile(self, account: str) -> Optional[dict]:
        cols = ", ".join(AGENT_ACCOUNT_COLUMNS)
        return self.conn.execute(
            f"SELECT {cols} FROM accounts WHERE account = %s",
            (account,),
        ).fetchone()

    def transactions(self, accounts: list[str], start: datetime, end: datetime, limit: int) -> pd.DataFrame:
        cols = ", ".join(TX_COLUMNS)
        rows = self.conn.execute(
            f"SELECT {cols} FROM transactions WHERE (src = ANY(%s) OR dst = ANY(%s)) "
            f"AND ts BETWEEN %s AND %s ORDER BY ts LIMIT %s",
            (accounts, accounts, start, end, limit),
        ).fetchall()
        if not rows:
            return pd.DataFrame(columns=TX_COLUMNS)
        df = pd.DataFrame(rows, columns=TX_COLUMNS)
        df["ts"] = pd.to_datetime(df["ts"])
        return df

    def search_policy(self, query: str, k: int = 3) -> list[PolicyChunk]:
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer(config.EMBED_MODEL)
        vec = self._embedder.encode(query, normalize_embeddings=True)
        rows = self.conn.execute(
            "SELECT doc_id, title, text FROM policy_chunks ORDER BY embedding <=> %s LIMIT %s",
            (vec, k),
        ).fetchall()
        return [{"doc_id": r["doc_id"], "title": r["title"], "text": r["text"]} for r in rows]


class PgStore:
    """ReviewStore + EvalStore over the OWNER connection. Never handed to ToolBox or triage()."""

    def __init__(self, conn):
        self.conn = conn

    def list_alerts(self, status: AlertStatus = "OPEN", limit: int = 50) -> list[dict]:
        lim = max(1, min(int(limit), 500))
        return self.conn.execute(
            "SELECT a.alert_id, a.score, a.score_rank, a.status, t.ts, t.src, t.dst, t.amount_usd, "
            "t.payment_format FROM alerts a JOIN transactions t USING (tx_id) "
            "WHERE a.status = %s ORDER BY a.score_rank LIMIT %s",
            (status, lim),
        ).fetchall()

    def latest_assessment(self, alert_id: int) -> Optional[dict]:
        return self.conn.execute(
            "SELECT * FROM assessments WHERE alert_id = %s ORDER BY assessment_id DESC LIMIT 1",
            (alert_id,),
        ).fetchone()

    def save_assessment(self, run: TriageRun) -> int:
        with self.conn.transaction():
            row = self.conn.execute(
                "INSERT INTO assessments (alert_id, model, final, llm_raw, guard_reasons, grounding, tool_log, "
                "latency_s, cost_usd) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING assessment_id",
                (
                    run["alert_id"],
                    run["model"],
                    json.dumps(run["final_assessment"], ensure_ascii=False, default=str),
                    json.dumps(run["llm_assessment"], ensure_ascii=False, default=str),
                    json.dumps(run["guard_reasons"], ensure_ascii=False, default=str),
                    json.dumps(run["grounding"], ensure_ascii=False, default=str),
                    json.dumps(run["tool_log"], ensure_ascii=False, default=str),
                    run["latency_s"],
                    run["cost_usd"],
                ),
            ).fetchone()
            self.conn.execute(
                "UPDATE alerts SET status = 'TRIAGED' WHERE alert_id = %s AND status = 'OPEN'",
                (run["alert_id"],),
            )
            return row["assessment_id"]

    def save_decision(self, alert_id: int, decision: Decision) -> int:
        with self.conn.transaction():
            latest = self.latest_assessment(alert_id)
            assessment_id = latest["assessment_id"] if latest else None
            row = self.conn.execute(
                "INSERT INTO decisions (alert_id, assessment_id, analyst, action, comment) "
                "VALUES (%s, %s, %s, %s, %s) RETURNING decision_id",
                (
                    alert_id,
                    assessment_id,
                    decision.analyst,
                    decision.action,
                    decision.comment,
                ),
            ).fetchone()
            self.conn.execute(
                "UPDATE alerts SET status = 'DECIDED' WHERE alert_id = %s",
                (alert_id,),
            )
            return row["decision_id"]

    def truth(self) -> pd.DataFrame:
        rows = self.conn.execute("SELECT alert_id, is_laundering, pattern FROM alert_truth").fetchall()
        return pd.DataFrame(rows)

    def set_memo(self, tx_id: int, memo: Optional[str]) -> None:
        self.conn.execute("UPDATE transactions SET memo = %s WHERE tx_id = %s", (memo, tx_id))
