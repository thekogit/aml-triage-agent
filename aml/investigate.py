"""Investigation functions behind the agent's tools. Pure Python over a read-only `Repo`, so they run
on in-memory frames (FrameRepo, tests) and on Postgres (db.PgRepo). Every function returns plain
JSON-serialisable data. NOTE: post-alert data is legitimate here (analyst review happens later);
the no-future rule applies to model features only."""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from .pii import Redactor
from .schema import TX_COLUMNS, AlertContext, PolicyChunk, Repo

__all__ = ["Repo", "FrameRepo", "alert_context", "account_profile", "recent_transactions",
           "counterparties", "find_cycles", "search_policy"]

MAX_CYCLES_RETURNED = 5
MAX_CYCLES_COLLECTED = 50
MAX_EXPANSIONS = 50_000


class FrameRepo:
    """In-memory Repo over pandas DataFrames (tests and offline demos). Same semantics as PgRepo:
    `transactions` filters src OR dst in accounts, start <= ts <= end (inclusive), ORDER BY ts, LIMIT."""

    def __init__(self, tx: pd.DataFrame, accounts: pd.DataFrame, alerts: pd.DataFrame,
                 policies: Optional[list[PolicyChunk]] = None):
        self.tx = tx.reindex(columns=TX_COLUMNS)
        self.accounts = accounts.drop(columns=["pesel"], errors="ignore").set_index("account")
        self.alerts, self.policies = alerts, list(policies or [])

    def alert(self, alert_id: int) -> Optional[dict]:
        a = self.alerts[self.alerts["alert_id"] == alert_id]
        if a.empty:
            return None
        row = a.iloc[0].to_dict()
        row.update(self.tx[self.tx["tx_id"] == row["tx_id"]].iloc[0].to_dict())
        return row

    def profile(self, account: str) -> Optional[dict]:
        if account not in self.accounts.index:
            return None
        return self.accounts.loc[account].to_dict() | {"account": account}

    def transactions(self, accounts: list[str], start: datetime, end: datetime, limit: int) -> pd.DataFrame:
        m = (self.tx["src"].isin(accounts) | self.tx["dst"].isin(accounts)) & self.tx["ts"].between(start, end)
        return self.tx[m].sort_values("ts", kind="stable").head(limit).reset_index(drop=True)

    def search_policy(self, query: str, k: int) -> list[PolicyChunk]:
        words = set(query.lower().split())
        return sorted(self.policies, key=lambda p: -len(words & set(p["text"].lower().split())))[:k]


def _usd(x) -> Optional[float]:
    return None if x is None or pd.isna(x) else round(float(x), 2)


def alert_context(repo: Repo, alert_id: int) -> AlertContext | dict:
    a = repo.alert(alert_id)
    if a is None:
        return {"error": f"alert {alert_id} not found"}
    return {
        "alert_id": int(alert_id), "tx_id": int(a["tx_id"]), "timestamp": str(pd.Timestamp(a["ts"])),
        "from_account": str(a["src"]), "to_account": str(a["dst"]),
        "amount_paid": _usd(a["amount_paid"]), "payment_currency": str(a["payment_currency"]),
        "amount_usd": _usd(a.get("amount_usd")), "payment_format": str(a["payment_format"]),
        "model_score": round(float(a["score"]), 4), "score_rank": int(a["score_rank"]),
        "score_rank_pct": float(a["score_rank_pct"]), "model_reasons": str(a["reasons"]),
    }


def account_profile(repo: Repo, account: str, redactor: Redactor) -> dict:
    """KYC profile with PESEL DROPPED (never needed) and every text field redacted."""
    p = repo.profile(account)
    if p is None:
        return {"account": account, "error": "no customer profile on file"}
    out: dict = {}
    for k, v in p.items():
        if k == "pesel":
            continue
        if v is None or (isinstance(v, float) and np.isnan(v)):
            out[k] = None
        elif isinstance(v, (bool, np.bool_)):
            out[k] = bool(v)
        elif isinstance(v, (int, np.integer)):
            out[k] = int(v)
        elif isinstance(v, (float, np.floating)):
            out[k] = float(v)
        else:
            out[k] = redactor.redact(str(v))[0]
    return out


def recent_transactions(repo: Repo, account: str, center: datetime, hours_before: int = 72,
                        hours_after: int = 24, limit: int = 40) -> dict:
    df = repo.transactions([account], center - timedelta(hours=hours_before),
                           center + timedelta(hours=hours_after), 5000)
    out = df[df["src"] == account]
    inc = df[df["dst"] == account]
    rows = [{
        "tx_id": int(r.tx_id), "ts": str(r.ts), "direction": "OUT" if r.src == account else "IN",
        "counterparty": r.dst if r.src == account else r.src, "amount_usd": _usd(r.amount_usd),
        "payment_format": r.payment_format, "memo": None if pd.isna(r.memo) else str(r.memo),
    } for r in df.head(limit).itertuples(index=False)]
    return {
        "account": account, "window": f"-{hours_before}h/+{hours_after}h around {center}",
        "n_out": int(len(out)), "usd_out": _usd(out["amount_usd"].sum()),
        "n_in": int(len(inc)), "usd_in": _usd(inc["amount_usd"].sum()),
        "distinct_payees": int(out["dst"].nunique()), "distinct_payers": int(inc["src"].nunique()),
        "transactions": rows, "truncated": bool(len(df) > limit),
    }


def counterparties(repo: Repo, account: str, center: datetime, hours: int = 72) -> dict:
    df = repo.transactions([account], center - timedelta(hours=hours), center + timedelta(hours=hours), 20000)

    def top(side: pd.DataFrame, key: str) -> tuple[list[dict], int]:
        g = side.groupby(key)["amount_usd"].agg(["count", "sum"])
        g = g.reset_index().sort_values(["sum", key], ascending=[False, True], kind="stable")
        return [{"counterparty": str(r[key]), "n": int(r["count"]), "usd": _usd(r["sum"])}
                for _, r in g.head(10).iterrows()], int(len(g))

    payees, n_payees = top(df[df["src"] == account], "dst")
    payers, n_payers = top(df[df["dst"] == account], "src")
    return {"account": account, "window_hours": hours, "payees": payees, "payers": payers,
            "n_payees": n_payees, "n_payers": n_payers}


def _temporal_cycles(edges: pd.DataFrame, account: str, max_len: int) -> tuple[list[dict], int, bool]:
    """Simple directed cycles through `account` with 2..max_len accounts whose hops are TIME-ORDERED:
    each hop uses the earliest transaction on that edge with ts >= previous hop's ts (greedy earliest
    arrival is exact for feasibility). Pruned by backward BFS distance to `account`; hard cap of
    MAX_EXPANSIONS edge expansions. Returns (top cycles by min_hop_usd desc, n_found, truncated)."""
    e = edges.loc[edges["src"] != edges["dst"], ["tx_id", "ts", "src", "dst", "amount_usd"]]
    e = e.drop_duplicates("tx_id")
    if e.empty:
        return [], 0, False
    e = e.sort_values(["src", "dst", "ts", "tx_id"], kind="stable")
    ts_ns = pd.to_datetime(e["ts"]).to_numpy(dtype="datetime64[ns]").astype(np.int64)
    usd = e["amount_usd"].to_numpy(dtype=np.float64)
    ids = e["tx_id"].to_numpy(dtype=np.int64)
    src = e["src"].to_numpy()
    dst = e["dst"].to_numpy()

    # adjacency: src -> list of (dst, ts[], usd[], ids[]) ordered by total usd desc, then dst
    bounds = np.flatnonzero((src[1:] != src[:-1]) | (dst[1:] != dst[:-1])) + 1
    starts = np.concatenate([[0], bounds])
    ends = np.concatenate([bounds, [len(e)]])
    adj: dict[str, list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]] = {}
    radj: dict[str, list[str]] = {}
    for s0, s1 in zip(starts, ends):
        u, v = str(src[s0]), str(dst[s0])
        adj.setdefault(u, []).append((v, ts_ns[s0:s1], usd[s0:s1], ids[s0:s1]))
        radj.setdefault(v, []).append(u)
    for u in adj:
        adj[u].sort(key=lambda x: (-float(np.nansum(x[2])), x[0]))

    dist = {account: 0}                       # min hops from node to `account`
    frontier = [account]
    for k in range(1, max_len):
        nxt = []
        for v in frontier:
            for u in radj.get(v, ()):
                if u not in dist:
                    dist[u] = k
                    nxt.append(u)
        frontier = nxt

    found: list[dict] = []
    state = {"expansions": 0, "truncated": False}

    def dfs(v: str, t_prev: int, path: list[str], hops: list[dict]) -> None:
        for w, ts, am, tx in adj.get(v, ()):
            if state["truncated"] or len(found) >= MAX_CYCLES_COLLECTED:
                return
            state["expansions"] += 1
            if state["expansions"] > MAX_EXPANSIONS:
                state["truncated"] = True
                return
            i = int(np.searchsorted(ts, t_prev, side="left"))
            if i >= len(ts):
                continue
            hop = {"from": v, "to": w, "tx_id": int(tx[i]), "ts": str(pd.Timestamp(int(ts[i]))), "usd": _usd(am[i])}
            if w == account:
                if len(path) >= 2:
                    found.append({"path": path + [account], "hops": hops + [hop]})
                continue
            if w in path or w not in dist or len(path) + dist[w] > max_len:
                continue
            dfs(w, int(ts[i]), path + [w], hops + [hop])

    dfs(account, np.iinfo(np.int64).min, [account], [])
    for c in found:
        amounts = [h["usd"] for h in c["hops"] if h["usd"] is not None]
        c["min_hop_usd"] = min(amounts) if amounts else None
        first, last = pd.Timestamp(c["hops"][0]["ts"]), pd.Timestamp(c["hops"][-1]["ts"])
        c["duration_hours"] = round((last - first).total_seconds() / 3600, 1)
    found.sort(key=lambda c: (-(c["min_hop_usd"] or 0.0), len(c["path"]), c["path"]))
    return found[:MAX_CYCLES_RETURNED], len(found), state["truncated"]


def find_cycles(repo: Repo, account: str, center: datetime, hours: int = 168, max_len: int = 4) -> dict:
    """Round-tripping check in [center - hours, center + hours]. Fetches 2-hop edges (txs touching the
    account's direct counterparties), which covers every cycle of length <= 4; max_len is clamped to 2..4."""
    max_len = max(2, min(int(max_len), 4))
    start, end = center - timedelta(hours=hours), center + timedelta(hours=hours)
    first = repo.transactions([account], start, end, 20_000)
    neighbours = sorted(set(first["src"]) | set(first["dst"]) | {account})
    edges = repo.transactions(neighbours, start, end, 200_000)
    cycles, n_found, truncated = _temporal_cycles(edges, account, max_len)
    return {"account": account, "window_hours": hours, "max_len": max_len, "cycles": cycles,
            "n_cycles_found": n_found, "search_truncated": truncated}


def search_policy(repo: Repo, query: str, k: int = 3) -> dict:
    return {"query": query, "results": [{"doc_id": p["doc_id"], "title": p["title"], "text": p["text"][:1200]}
                                        for p in repo.search_policy(query, k)]}
