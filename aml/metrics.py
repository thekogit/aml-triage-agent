"""Alert-tuning metrics: PR-AUC, F1 at a validation threshold, and precision/recall at an alert budget."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve


def best_f1_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """Score threshold maximising F1 (fit on validation days ONLY). prec/rec have len(thr)+1 entries;
    the last (prec=1, rec=0) point has no threshold, hence f1[:-1]."""
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    return float(thr[int(np.argmax(f1[:-1]))])


def at_budget(y: np.ndarray, p: np.ndarray, budget: float) -> dict:
    """Alert on the top ceil(budget * n) rows by score (stable order on ties)."""
    k = max(1, math.ceil(budget * len(p)))
    top = np.argsort(-p, kind="stable")[:k]
    tp = int(y[top].sum())
    return {"alerts": k, "precision_at_budget": tp / k, "recall_at_budget": tp / max(1, int(y.sum()))}


def summary(y: np.ndarray, p: np.ndarray, threshold: float, budget: float) -> dict:
    pred = (p >= threshold).astype(int)
    out = {
        "n": int(len(y)),
        "positives": int(y.sum()),
        "pr_auc": round(float(average_precision_score(y, p)), 4),
        "f1_at_val_threshold": round(float(f1_score(y, pred, zero_division=0)), 4),
    }
    out.update({k: (round(v, 4) if isinstance(v, float) else v) for k, v in at_budget(y, p, budget).items()})
    return out


def reliability_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Mean predicted score vs observed positive rate per score decile (rank-based equal-count bins)."""
    df = pd.DataFrame({"y": y, "p": p})
    df["bin"] = pd.qcut(df["p"].rank(method="first"), q=bins, labels=False)
    return df.groupby("bin").agg(mean_score=("p", "mean"), observed_rate=("y", "mean"), n=("y", "size")).reset_index()
