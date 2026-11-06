"""Leakage-free time-window aggregates in O(n log n) with numpy (no Python loop over groups).

Sort events by (group, time) once, place them on one number line key = group*scale + time,
and answer every query with two binary searches over a prefix sum.
"""
from __future__ import annotations

import numpy as np


def window_count_sum(
    ev_group: np.ndarray, ev_time: np.ndarray, ev_value: np.ndarray,
    q_group: np.ndarray, q_time: np.ndarray, window_s: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Count and sum of events with ev_group == q_group and ev_time in (q_time - window_s, q_time].

    Groups: non-negative int codes. Times: non-negative int seconds. NaN values add 0 to the sum.
    Queries with a negative group code (unknown group) get count 0 and sum 0.0.
    The window is right-closed: events at exactly q_time ARE included (same-timestamp visibility).
    """
    ev_group = np.asarray(ev_group, dtype=np.int64)
    ev_time = np.asarray(ev_time, dtype=np.int64)
    ev_value = np.asarray(ev_value, dtype=np.float64)
    q_group = np.asarray(q_group, dtype=np.int64)
    q_time = np.asarray(q_time, dtype=np.int64)
    if window_s <= 0:
        raise ValueError("window_s must be positive")

    max_time = int(max(ev_time.max(initial=0), q_time.max(initial=0)))
    scale = max_time + int(window_s) + 1  # neighbouring groups can never overlap on the number line

    order = np.lexsort((ev_time, ev_group))
    keys = ev_group[order] * scale + ev_time[order]
    csum = np.concatenate([[0.0], np.cumsum(np.nan_to_num(ev_value[order]))])

    q_key = q_group * scale + q_time
    hi = np.searchsorted(keys, q_key, side="right")
    lo = np.searchsorted(keys, q_key - window_s, side="right")
    count = (hi - lo).astype(np.int64)
    total = csum[hi] - csum[lo]
    unknown = q_group < 0
    count[unknown] = 0
    total[unknown] = 0.0
    return count, total
