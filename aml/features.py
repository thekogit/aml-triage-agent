"""Per-transaction features. Every feature uses only transactions at or before the row's timestamp."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .windows import window_count_sum

H24, D7 = 24 * 3600, 7 * 24 * 3600
CATEGORICAL = ["payment_format", "payment_currency", "receiving_currency"]
NUMERIC = [
    "log_amount_paid", "log_amount_usd", "cross_currency", "self_transfer", "hour", "dow",
    "src_out_cnt_24h", "src_out_usd_24h", "src_out_cnt_7d",
    "dst_in_cnt_24h", "dst_in_usd_24h", "dst_in_cnt_7d",
    "src_in_usd_24h", "passthrough_ratio", "dst_out_cnt_24h",
    "reverse_pair_cnt_7d", "src_distinct_dst_7d", "dst_distinct_src_7d",
]
FEATURES = NUMERIC + CATEGORICAL
FORBIDDEN_INPUTS = ("is_laundering", "pattern", "attempt_id", "split")  # never used to compute a feature


def usd_rates(tx: pd.DataFrame, usd_name: str, fit_mask: Optional[pd.Series] = None) -> pd.Series:
    """USD per unit of each currency = median(amount_received / amount_paid) over rows paid in that
    currency and received in USD. `fit_mask` restricts the rows used (train days only in production).
    Currencies never observed get no entry (-> NaN amount_usd downstream)."""
    m = (tx["receiving_currency"] == usd_name) & (tx["payment_currency"] != usd_name) & (tx["amount_paid"] > 0)
    if fit_mask is not None:
        m &= fit_mask.to_numpy()
    rates = (tx.loc[m, "amount_received"] / tx.loc[m, "amount_paid"]).groupby(tx.loc[m, "payment_currency"]).median()
    rates[usd_name] = 1.0
    return rates


def _daily_distinct(codes_a: np.ndarray, codes_b: np.ndarray, day: np.ndarray, q_codes: np.ndarray,
                    q_day: np.ndarray) -> np.ndarray:
    """Sum over the previous 7 COMPLETED calendar days of the number of distinct b per (a, day).
    A finished day becomes visible at the next midnight, so the current day never counts."""
    d = pd.DataFrame({"a": codes_a, "b": codes_b, "day": day}).drop_duplicates()
    per_day = d.groupby(["a", "day"]).size().reset_index(name="n")
    stamp = (per_day["day"].to_numpy() + 1) * 86400
    _, total = window_count_sum(per_day["a"].to_numpy(), stamp, per_day["n"].to_numpy(),
                                q_codes, q_day * 86400, D7)
    return total


def build_features(tx: pd.DataFrame, usd_name: str = "US Dollar") -> pd.DataFrame:
    """tx columns: tx_id, ts, src, dst, amount_paid, payment_currency, amount_received,
    receiving_currency, payment_format, is_laundering, split (pattern optional).
    Returns one row per tx (same order) with FEATURES + tx_id, amount_usd, is_laundering, split[, pattern]."""
    base = tx["ts"].min().normalize()
    t = ((tx["ts"] - base).dt.total_seconds()).to_numpy(dtype=np.int64)
    day = t // 86400

    codes, _ = pd.factorize(pd.concat([tx["src"], tx["dst"]], ignore_index=True))
    n = len(tx)
    src, dst = codes[:n].astype(np.int64), codes[n:].astype(np.int64)

    fit = (tx["split"] == "train") if "split" in tx else None
    if fit is not None and not bool(fit.any()):
        fit = None
    rates = usd_rates(tx, usd_name, fit)
    usd = (tx["amount_paid"] * tx["payment_currency"].map(rates)).to_numpy(dtype=np.float64)
    usd_filled = np.nan_to_num(usd)
    tx_id = tx["tx_id"].to_numpy()

    f = pd.DataFrame({"tx_id": tx_id})
    f["log_amount_paid"] = np.log1p(tx["amount_paid"].to_numpy())
    f["log_amount_usd"] = np.log1p(usd)
    f["cross_currency"] = (tx["payment_currency"] != tx["receiving_currency"]).astype(int).to_numpy()
    f["self_transfer"] = (src == dst).astype(int)
    f["hour"] = tx["ts"].dt.hour.to_numpy()
    f["dow"] = tx["ts"].dt.day_of_week.to_numpy()

    f["src_out_cnt_24h"], f["src_out_usd_24h"] = window_count_sum(src, t, usd_filled, src, t, H24, ev_id=tx_id, q_id=tx_id)
    f["src_out_cnt_7d"], _ = window_count_sum(src, t, usd_filled, src, t, D7, ev_id=tx_id, q_id=tx_id)
    f["dst_in_cnt_24h"], f["dst_in_usd_24h"] = window_count_sum(dst, t, usd_filled, dst, t, H24, ev_id=tx_id, q_id=tx_id)
    f["dst_in_cnt_7d"], _ = window_count_sum(dst, t, usd_filled, dst, t, D7, ev_id=tx_id, q_id=tx_id)
    # money that flowed INTO the sender in the last 24h (layering / pass-through signal)
    _, f["src_in_usd_24h"] = window_count_sum(dst, t, usd_filled, src, t, H24, ev_id=tx_id, q_id=tx_id)
    f["passthrough_ratio"] = np.where(f["src_in_usd_24h"] > 0, usd_filled / (f["src_in_usd_24h"] + 1e-9), 0.0)
    # how often the receiver itself sent money out in the last 24h
    f["dst_out_cnt_24h"], _ = window_count_sum(src, t, usd_filled, dst, t, H24, ev_id=tx_id, q_id=tx_id)

    # did the receiver pay the sender in the last 7 days? (2-cycles / round-tripping)
    n_acc = int(max(src.max(), dst.max())) + 1
    forward = src * n_acc + dst
    reverse = dst * n_acc + src
    pair_index = pd.Index(pd.unique(forward))
    f_code = pair_index.get_indexer(forward)
    r_code = pair_index.get_indexer(reverse)  # -1 when the reverse pair never occurs
    f["reverse_pair_cnt_7d"], _ = window_count_sum(f_code, t, np.ones(n), r_code, t, D7, ev_id=tx_id, q_id=tx_id)

    f["src_distinct_dst_7d"] = _daily_distinct(src, dst, day, src, day)
    f["dst_distinct_src_7d"] = _daily_distinct(dst, src, day, dst, day)

    for col in CATEGORICAL:
        f[col] = tx[col].astype(str).to_numpy()
    f["amount_usd"] = usd
    f["is_laundering"] = tx["is_laundering"].to_numpy()
    f["split"] = tx["split"].to_numpy()
    if "pattern" in tx:
        f["pattern"] = tx["pattern"].to_numpy()
    return f
