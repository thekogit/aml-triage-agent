"""Read the IBM AMLworld Kaggle files into one clean transaction table with a whole-day temporal split."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# The Kaggle CSV has two columns both named "Account"; we name all 11 columns ourselves.
RAW_COLUMNS = [
    "timestamp", "from_bank", "from_account", "to_bank", "to_account",
    "amount_received", "receiving_currency", "amount_paid", "payment_currency",
    "payment_format", "is_laundering",
]
EXPECTED_HEADER = ("Timestamp,From Bank,Account,To Bank,Account,Amount Received,Receiving Currency,"
                   "Amount Paid,Payment Currency,Payment Format,Is Laundering")
STR_COLS = ["from_bank", "from_account", "to_bank", "to_account"]
TS_FORMAT = "%Y/%m/%d %H:%M"
PATTERN_KEY = ["ts", "src", "dst", "amount_paid", "payment_format"]


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df["ts"] = pd.to_datetime(df["timestamp"], format=TS_FORMAT)
    df["src"] = df["from_bank"].str.strip() + "_" + df["from_account"].str.strip()
    df["dst"] = df["to_bank"].str.strip() + "_" + df["to_account"].str.strip()
    for col in ["receiving_currency", "payment_currency", "payment_format"]:
        df[col] = df[col].astype(str).str.strip()
    df["is_laundering"] = df["is_laundering"].astype(int)
    return df[["ts", "src", "dst", "amount_paid", "payment_currency", "amount_received",
               "receiving_currency", "payment_format", "is_laundering"]]


def read_transactions(path: Path, nrows: Optional[int] = None) -> pd.DataFrame:
    """Kaggle CSV -> columns tx_id, ts, src, dst, amount_paid, payment_currency, amount_received,
    receiving_currency, payment_format, is_laundering. tx_id = row index after a stable time sort."""
    df = pd.read_csv(path, header=0, names=RAW_COLUMNS, dtype={c: str for c in STR_COLS}, nrows=nrows)
    df = _clean(df)
    df = df.sort_values(["ts", "src", "dst", "amount_paid"], kind="stable").reset_index(drop=True)
    df.insert(0, "tx_id", np.arange(len(df), dtype=np.int64))
    return df


def read_patterns(path: Path) -> pd.DataFrame:
    """Parse *_Patterns.txt: blocks between 'BEGIN LAUNDERING ATTEMPT - <NAME>[: details]' and
    'END LAUNDERING ATTEMPT ...'. Returns ts, src, dst, amount_paid, payment_format, pattern, attempt_id."""
    rows, attempt, name = [], 0, None
    with open(path, encoding="utf-8-sig") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("BEGIN LAUNDERING ATTEMPT"):
                attempt += 1
                label = line.split("-", 1)[1] if "-" in line else "UNKNOWN"
                name = label.split(":")[0].strip().upper()
            elif line.startswith("END LAUNDERING ATTEMPT"):
                name = None
            elif name is not None:
                fields = next(csv.reader([line]))
                if len(fields) >= 11:
                    rows.append(fields[:11] + [name, attempt])
    df = pd.DataFrame(rows, columns=RAW_COLUMNS + ["pattern", "attempt_id"])
    df["amount_paid"] = df["amount_paid"].astype(float)
    df["amount_received"] = df["amount_received"].astype(float)
    df = _clean(df).assign(pattern=df["pattern"].values, attempt_id=df["attempt_id"].values)
    return df[PATTERN_KEY + ["pattern", "attempt_id"]]


def attach_patterns(tx: pd.DataFrame, patterns: pd.DataFrame) -> pd.DataFrame:
    """Left-join pattern labels on PATTERN_KEY. Non-laundering -> 'NONE'; laundering without a
    pattern row -> 'UNCLASSIFIED'. Row count and order of `tx` are preserved."""
    p = patterns.drop_duplicates(subset=PATTERN_KEY)
    out = tx.merge(p, on=PATTERN_KEY, how="left")
    out["pattern"] = out["pattern"].fillna("NONE")
    out.loc[out["is_laundering"] == 0, "pattern"] = "NONE"
    out.loc[(out["is_laundering"] == 1) & (out["pattern"] == "NONE"), "pattern"] = "UNCLASSIFIED"
    return out


def temporal_split(ts: pd.Series, fractions: tuple[float, float, float] = (0.6, 0.2, 0.2)) -> pd.Series:
    """Assign whole days to train/val/test so the row shares are closest to `fractions`
    (minimise the max relative error; IBM Multi-GNN rule). Vectorised over all (i, j) cut pairs:
    train = days < i, val = i <= day < j, test = day >= j, with 1 <= i < j <= n_days - 1.
    Ties resolve to the lexicographically first (i, j). Needs >= 3 distinct calendar days."""
    day = (ts - ts.min().normalize()).dt.days.to_numpy()
    counts = np.bincount(day).astype(np.float64)
    n_days = len(counts)
    if n_days < 3:
        raise ValueError("temporal_split needs at least 3 calendar days")
    c = np.concatenate([[0.0], np.cumsum(counts)])
    total = c[-1]
    frac = np.asarray(fractions, dtype=np.float64)
    i, j = np.triu_indices(n_days, k=1)          # row-major order == nested-loop order
    keep = (i >= 1) & (j <= n_days - 1)
    i, j = i[keep], j[keep]
    parts = np.stack([c[i], c[j] - c[i], total - c[j]], axis=1) / total
    err = np.max(np.abs(parts - frac) / frac, axis=1)
    k = int(np.argmin(err))
    cut_i, cut_j = int(i[k]), int(j[k])
    split = np.where(day < cut_i, "train", np.where(day < cut_j, "val", "test"))
    return pd.Series(split, index=ts.index, name="split")
