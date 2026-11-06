"""Postgres + pgvector serving store. Two classes, two roles:
PgRepo  -> Repo protocol, connects as SELECT-only role `aml_agent` (cannot see alert_truth or pesel).
PgStore -> ReviewStore + EvalStore, connects as owner `aml`. The only writers in the online system."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

import pandas as pd

from . import config
from .schema import TX_COLUMNS, AlertStatus, Decision, PolicyChunk, TriageRun

AGENT_ACCOUNT_COLUMNS = ["account", "customer_type", "name", "iban", "risk_rating",
                         "declared_monthly_turnover_usd", "onboarding_date", "country"]  # no pesel

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
-- audit trail is append-only (TRUNCATE by the loader is still allowed)
CREATE OR REPLACE RULE assessments_no_update AS ON UPDATE TO assessments DO INSTEAD NOTHING;
CREATE OR REPLACE RULE assessments_no_delete AS ON DELETE TO assessments DO INSTEAD NOTHING;
CREATE OR REPLACE RULE decisions_no_update AS ON UPDATE TO decisions DO INSTEAD NOTHING;
CREATE OR REPLACE RULE decisions_no_delete AS ON DELETE TO decisions DO INSTEAD NOTHING;
-- least-privilege role for the agent: no alert_truth, no pesel, no writes
DO $$ BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'aml_agent') THEN
    CREATE ROLE aml_agent LOGIN PASSWORD 'aml_agent';
  END IF;
END $$;
REVOKE ALL ON ALL TABLES IN SCHEMA public FROM aml_agent;
GRANT SELECT ON transactions, alerts, policy_chunks TO aml_agent;
GRANT SELECT ({", ".join(AGENT_ACCOUNT_COLUMNS)}) ON accounts TO aml_agent;
"""


def connect(dsn: str = config.PG_DSN):
    """psycopg.connect(dsn, autocommit=True, row_factory=dict_row) + pgvector register_vector(conn).

    # TODO(builder): import psycopg / dict_row / pgvector.psycopg.register_vector INSIDE the function
    #   (CI has no psycopg). Do NOT run CREATE EXTENSION here (agent role lacks rights); init_schema does it.
    """
    raise NotImplementedError


def init_schema(conn) -> None:
    """conn.execute(SCHEMA_SQL). Owner connection only. Idempotent.
    # TODO(builder): one line.
    """
    raise NotImplementedError


def copy_frame(conn, table: str, df: pd.DataFrame, columns: list[str]) -> int:
    """Bulk insert with COPY; NaN/NaT -> None; returns row count.
    SQL: COPY {table} ({cols}) FROM STDIN  via  cur.copy(...) as cp: cp.write_row(row)
    # TODO(builder): clean = df[columns].astype(object).where(df[columns].notna(), None);
    #   numpy scalars -> python via .item() if hasattr(v, "item"). `table` comes from code, never user input.
    """
    raise NotImplementedError


class PgRepo:
    """Read-only Repo over Postgres. MUST be constructed with connect(config.PG_AGENT_DSN)."""

    def __init__(self, conn, embedder=None):
        self.conn, self._embedder = conn, embedder

    def alert(self, alert_id: int) -> Optional[dict]:
        """SQL: SELECT a.alert_id, a.score, a.score_rank, a.score_rank_pct, a.reasons::text AS reasons, t.*
                FROM alerts a JOIN transactions t USING (tx_id) WHERE a.alert_id = %s   -> fetchone() or None
        # TODO(builder): one execute; parameterised; returns dict_row.
        """
        raise NotImplementedError

    def profile(self, account: str) -> Optional[dict]:
        """SQL: SELECT {AGENT_ACCOUNT_COLUMNS} FROM accounts WHERE account = %s  -> fetchone() or None
        # TODO(builder): explicit column list from AGENT_ACCOUNT_COLUMNS (SELECT * fails: column grant).
        """
        raise NotImplementedError

    def transactions(self, accounts: list[str], start: datetime, end: datetime, limit: int) -> pd.DataFrame:
        """SQL: SELECT {TX_COLUMNS} FROM transactions WHERE (src = ANY(%s) OR dst = ANY(%s))
                AND ts BETWEEN %s AND %s ORDER BY ts LIMIT %s
        Returns DataFrame with columns == TX_COLUMNS even when empty.
        # TODO(builder): pd.DataFrame(rows, columns=TX_COLUMNS); ts stays datetime64.
        """
        raise NotImplementedError

    def search_policy(self, query: str, k: int) -> list[PolicyChunk]:
        """SQL: SELECT doc_id, title, text FROM policy_chunks ORDER BY embedding <=> %s LIMIT %s
        vec = embedder.encode(query, normalize_embeddings=True); embedder lazily =
        SentenceTransformer(config.EMBED_MODEL) on first call.
        # TODO(builder): lazy import sentence_transformers inside the method.
        """
        raise NotImplementedError


class PgStore:
    """ReviewStore + EvalStore over the OWNER connection. Never handed to ToolBox or triage()."""

    def __init__(self, conn):
        self.conn = conn

    def list_alerts(self, status: AlertStatus = "OPEN", limit: int = 50) -> list[dict]:
        """SQL: SELECT a.alert_id, a.score, a.score_rank, a.status, t.ts, t.src, t.dst, t.amount_usd,
                t.payment_format FROM alerts a JOIN transactions t USING (tx_id)
                WHERE a.status = %s ORDER BY a.score_rank LIMIT %s
        # TODO(builder): clamp limit to 1..500.
        """
        raise NotImplementedError

    def latest_assessment(self, alert_id: int) -> Optional[dict]:
        """SQL: SELECT * FROM assessments WHERE alert_id = %s ORDER BY assessment_id DESC LIMIT 1
        # TODO(builder): fetchone().
        """
        raise NotImplementedError

    def save_assessment(self, run: TriageRun) -> int:
        """INSERT INTO assessments (alert_id, model, final, llm_raw, guard_reasons, grounding, tool_log,
           latency_s, cost_usd) VALUES (...) RETURNING assessment_id;
           then UPDATE alerts SET status = 'TRIAGED' WHERE alert_id = %s AND status = 'OPEN'
        JSONB values via json.dumps(..., ensure_ascii=False, default=str). Never sets DECIDED.
        # TODO(builder): wrap both statements in `with self.conn.transaction():`.
        """
        raise NotImplementedError

    def save_decision(self, alert_id: int, decision: Decision) -> int:
        """INSERT INTO decisions (alert_id, assessment_id, analyst, action, comment) VALUES (...)
           RETURNING decision_id; then UPDATE alerts SET status = 'DECIDED' WHERE alert_id = %s
        The ONLY code path that sets DECIDED. `decision` is already validated (Decision model).
        # TODO(builder): wrap both statements in `with self.conn.transaction():`.
        """
        raise NotImplementedError

    def truth(self) -> pd.DataFrame:
        """SQL: SELECT alert_id, is_laundering, pattern FROM alert_truth  -> DataFrame (eval only).
        # TODO(builder): one line.
        """
        raise NotImplementedError

    def set_memo(self, tx_id: int, memo: Optional[str]) -> None:
        """SQL: UPDATE transactions SET memo = %s WHERE tx_id = %s   (injection test only).
        # TODO(builder): one line.
        """
        raise NotImplementedError
