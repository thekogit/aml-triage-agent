"""Leakage-free time-window aggregates in O(n log n) with numpy (no Python loop over groups).

Sort events by (group, time, id) once, and answer every query with two binary searches over a prefix sum.
Within the same timestamp, queries only see transactions with tx_id <= query tx_id.
"""
from __future__ import annotations

import numpy as np


def window_count_sum(
    ev_group: np.ndarray,
    ev_time: np.ndarray,
    ev_value: np.ndarray,
    q_group: np.ndarray,
    q_time: np.ndarray,
    window_s: int,
    ev_id: np.ndarray | None = None,
    q_id: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Count and sum of events with ev_group == q_group and ev_time in (q_time - window_s, q_time].

    Groups: non-negative int codes. Times: non-negative int seconds. NaN values add 0 to the sum.
    Queries with a negative group code (unknown group) get count 0 and sum 0.0.
    The window is right-closed up to q_id: events at exactly q_time are included only if ev_id <= q_id.
    """
    ev_group = np.asarray(ev_group, dtype=np.int64)
    ev_time = np.asarray(ev_time, dtype=np.int64)
    ev_value = np.asarray(ev_value, dtype=np.float64)
    q_group = np.asarray(q_group, dtype=np.int64)
    q_time = np.asarray(q_time, dtype=np.int64)
    if window_s <= 0:
        raise ValueError("window_s must be positive")

    n_ev = len(ev_group)
    n_q = len(q_group)

    if ev_id is None:
        ev_id_arr = np.zeros(n_ev, dtype=np.int64)
    else:
        ev_id_arr = np.asarray(ev_id, dtype=np.int64)

    if q_id is None:
        q_id_arr = np.full(n_q, np.iinfo(np.int64).max, dtype=np.int64)
    else:
        q_id_arr = np.asarray(q_id, dtype=np.int64)

    dt = np.dtype([("g", np.int64), ("t", np.int64), ("id", np.int64)])
    order = np.lexsort((ev_id_arr, ev_time, ev_group))

    ev_sorted = np.empty(n_ev, dtype=dt)
    ev_sorted["g"] = ev_group[order]
    ev_sorted["t"] = ev_time[order]
    ev_sorted["id"] = ev_id_arr[order]
    csum = np.concatenate([[0.0], np.cumsum(np.nan_to_num(ev_value[order]))])

    q_hi = np.empty(n_q, dtype=dt)
    q_hi["g"] = q_group
    q_hi["t"] = q_time
    q_hi["id"] = q_id_arr

    q_lo = np.empty(n_q, dtype=dt)
    q_lo["g"] = q_group
    q_lo["t"] = q_time - int(window_s)
    q_lo["id"] = np.iinfo(np.int64).max

    hi = np.searchsorted(ev_sorted, q_hi, side="right")
    lo = np.searchsorted(ev_sorted, q_lo, side="right")

    count = (hi - lo).astype(np.int64)
    total = csum[hi] - csum[lo]

    unknown = q_group < 0
    count[unknown] = 0
    total[unknown] = 0.0
    return count, total
