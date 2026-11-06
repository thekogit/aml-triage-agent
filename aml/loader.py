"""Choose the serving subset of transactions and load everything into Postgres."""
from __future__ import annotations

from datetime import timedelta

import pandas as pd

from . import db
from .pii import synthetic_profiles


def context_transactions(tx: pd.DataFrame, alerts: pd.DataFrame, lookback_days: int = 7) -> pd.DataFrame:
    """All transactions of alerted accounts and their direct counterparties, from `lookback_days`
    before the test period to its end. That is enough for 2-hop questions and cycles."""
    alerted = tx[tx["tx_id"].isin(alerts["tx_id"])]
    core = set(alerted["src"]) | set(alerted["dst"])
    test_start = tx.loc[tx["split"] == "test", "ts"].min()
    window = tx[tx["ts"] >= test_start - timedelta(days=lookback_days)]
    touching = window[window["src"].isin(core) | window["dst"].isin(core)]
    neighbours = set(touching["src"]) | set(touching["dst"])
    return window[window["src"].isin(neighbours) | window["dst"].isin(neighbours)].copy()


def load_all(conn, tx: pd.DataFrame, amount_usd: pd.Series, alerts: pd.DataFrame, truth: pd.DataFrame) -> dict:
    db.init_schema(conn)
    for table in ["decisions", "assessments", "alert_truth", "alerts", "transactions", "accounts"]:
        conn.execute(f"TRUNCATE {table} CASCADE")
    ctx = context_transactions(tx, alerts)
    ctx["amount_usd"] = amount_usd.reindex(ctx["tx_id"]).to_numpy()
    ctx["memo"] = None
    cols = ["tx_id", "ts", "src", "dst", "amount_paid", "payment_currency", "amount_received",
            "receiving_currency", "payment_format", "amount_usd", "memo"]
    n_tx = db.copy_frame(conn, "transactions", ctx, cols)
    profiles = synthetic_profiles(set(ctx["src"]) | set(ctx["dst"]))
    n_acc = db.copy_frame(conn, "accounts", profiles, list(profiles.columns))
    n_al = db.copy_frame(conn, "alerts", alerts.assign(status="OPEN"),
                         ["alert_id", "tx_id", "score", "score_rank", "score_rank_pct", "reasons", "status"])
    t = truth.merge(alerts[["alert_id", "tx_id"]], on="tx_id")
    if "pattern" not in t.columns:
        t["pattern"] = None
    db.copy_frame(conn, "alert_truth", t, ["alert_id", "is_laundering", "pattern"])
    return {"transactions": n_tx, "accounts": n_acc, "alerts": n_al}
