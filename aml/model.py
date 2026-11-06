"""Train the transaction-monitoring model, compare it with rules, and create alerts with reasons."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .features import CATEGORICAL, FEATURES
from .metrics import best_f1_threshold, reliability_table, summary


def rules_score(feat: pd.DataFrame) -> np.ndarray:
    """Baseline: number of red-flag rules hit (0..4) + log_amount_usd/100 as tie-breaker (< 1)."""
    fan_out = (feat["src_out_cnt_24h"] >= 5).astype(int)
    fan_in = (feat["dst_in_cnt_24h"] >= 5).astype(int)
    pass_through = ((feat["src_in_usd_24h"] > 0) & feat["passthrough_ratio"].between(0.8, 1.0)).astype(int)
    round_trip = (feat["reverse_pair_cnt_7d"] > 0).astype(int)
    flags = (fan_out + fan_in + pass_through + round_trip).to_numpy(dtype=np.float64)
    return flags + feat["log_amount_usd"].fillna(0).to_numpy(dtype=np.float64) / 100


def _plain(v):
    if isinstance(v, (np.floating, float)):
        return None if np.isnan(v) else round(float(v), 4)
    if isinstance(v, (np.integer, int)):
        return int(v)
    return str(v)


def top_reasons(contrib_row: np.ndarray, x_row: pd.Series, n: int = 5) -> list[dict]:
    """Top-n features by |SHAP contribution|. contrib_row excludes the bias column and is aligned
    with FEATURES (== X column order)."""
    idx = np.argsort(-np.abs(contrib_row), kind="stable")[:n]
    return [{"feature": FEATURES[i], "value": _plain(x_row.iloc[i]), "contribution": round(float(contrib_row[i]), 4)}
            for i in idx]


def to_matrix(feat: pd.DataFrame, categories: dict[str, list[str]]) -> pd.DataFrame:
    """feat[FEATURES] copy with CATEGORICAL columns cast to pd.Categorical(categories=categories[col]).

    Unseen categories -> NaN (LightGBM treats as missing). Column order MUST equal FEATURES.
    """
    X = feat[FEATURES].copy()
    for col in CATEGORICAL:
        X[col] = pd.Categorical(X[col], categories=categories[col])
    return X


def train(feat: pd.DataFrame, seed: int = 7):
    """Fit LightGBM on split=='train', early-stop on split=='val'. Returns (clf, categories)."""
    import lightgbm as lgb

    tr, va = feat[feat["split"] == "train"], feat[feat["split"] == "val"]
    categories = {c: sorted(tr[c].astype(str).unique().tolist()) for c in CATEGORICAL}
    clf = lgb.LGBMClassifier(
        n_estimators=3000,
        learning_rate=0.03,
        num_leaves=63,
        min_child_samples=50,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=seed,
        verbose=-1,
    )
    clf.fit(
        to_matrix(tr, categories),
        tr["is_laundering"],
        eval_set=[(to_matrix(va, categories), va["is_laundering"])],
        eval_metric="average_precision",
        callbacks=[lgb.early_stopping(200), lgb.log_evaluation(100)],
    )
    return clf, categories


def evaluate_and_alert(feat: pd.DataFrame, clf, categories: dict) -> dict:
    """Returns {"report", "alerts", "truth", "scores", "reliability"}."""
    va, te = feat[feat["split"] == "val"], feat[feat["split"] == "test"].reset_index(drop=True)
    p_val = clf.predict_proba(to_matrix(va, categories))[:, 1]
    thr = best_f1_threshold(va["is_laundering"].to_numpy(), p_val)
    X_te = to_matrix(te, categories)
    p_te = clf.predict_proba(X_te)[:, 1]
    y_te = te["is_laundering"].to_numpy()

    # Rules baseline evaluated at its best validation F1 threshold rather than np.inf
    r_val = rules_score(va)
    r_thr = best_f1_threshold(va["is_laundering"].to_numpy(), r_val)
    r_te = rules_score(te)

    report = {
        "model_test": summary(y_te, p_te, thr, config.ALERT_BUDGET),
        "rules_test": summary(y_te, r_te, r_thr, config.ALERT_BUDGET),
        "val_threshold": thr,
        "rules_val_threshold": r_thr,
        "best_iteration": int(getattr(clf, "best_iteration_", None) or clf.n_estimators),
    }

    k = report["model_test"]["alerts"]
    order = np.argsort(-p_te, kind="stable")
    top = order[:k]
    contrib = clf.booster_.predict(X_te.iloc[top], pred_contrib=True)[:, :-1]

    alerts = te.iloc[top][["tx_id"]].reset_index(drop=True)
    alerts["score"] = p_te[top]
    alerts["score_rank"] = np.arange(1, k + 1)
    alerts["score_rank_pct"] = alerts["score_rank"] / len(te)
    alerts["reasons"] = [
        json.dumps(top_reasons(row, X_te.iloc[i]), ensure_ascii=False)
        for row, i in zip(contrib, top)
    ]
    alerts.insert(0, "alert_id", np.arange(1, k + 1))

    truth_cols = ["tx_id", "is_laundering"]
    if "pattern" in te.columns:
        truth_cols.append("pattern")
    truth = te.iloc[top][truth_cols].reset_index(drop=True)
    truth.insert(0, "alert_id", np.arange(1, k + 1))

    scores = pd.DataFrame({"tx_id": te["tx_id"], "score": p_te, "is_laundering": y_te})
    return {
        "report": report,
        "alerts": alerts,
        "truth": truth,
        "scores": scores,
        "reliability": reliability_table(y_te, p_te),
    }


def save_report(out: dict, results_dir: Path) -> None:
    """Write results/model_eval.md and results/reliability_test.csv."""
    results_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {"model": name, **vals}
        for name, vals in [
            ("LightGBM", out["report"]["model_test"]),
            ("Rules baseline", out["report"]["rules_test"]),
        ]
    ]
    table = pd.DataFrame(rows)
    cols = list(table.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in r) + " |" for r in table.itertuples(index=False)]
    text = "\n".join(lines) + f"\n\nValidation-chosen threshold: {out['report']['val_threshold']:.4f}\n"
    (results_dir / "model_eval.md").write_text(text, encoding="utf-8")
    out["reliability"].to_csv(results_dir / "reliability_test.csv", index=False)
