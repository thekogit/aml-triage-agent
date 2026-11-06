"""Train the transaction-monitoring model, compare it with rules, and create alerts with reasons."""
from __future__ import annotations

import json  # noqa: F401  (used once stubs are implemented)
from pathlib import Path

import numpy as np
import pandas as pd

from . import config  # noqa: F401
from .features import CATEGORICAL, FEATURES  # noqa: F401
from .metrics import best_f1_threshold, reliability_table, summary  # noqa: F401


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
    # TODO(builder): X = feat[FEATURES].copy(); for col in CATEGORICAL: X[col] = pd.Categorical(X[col],
    #   categories=categories[col]); return X
    """
    raise NotImplementedError


def train(feat: pd.DataFrame, seed: int = 7):
    """Fit LightGBM on split=='train', early-stop on split=='val'. Returns (clf, categories).

    categories = {c: sorted(train[c].astype(str).unique())} for c in CATEGORICAL.
    lib: lgb.LGBMClassifier(n_estimators=3000, learning_rate=0.03, num_leaves=63, min_child_samples=50,
         subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=seed, verbose=-1)
         .fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="average_precision",
              callbacks=[lgb.early_stopping(200), lgb.log_evaluation(100)])
    # TODO(builder): `import lightgbm as lgb` INSIDE the function (CI has no lightgbm). Label column is
    #   is_laundering. Never pass split/pattern/is_laundering/amount_usd as features (only FEATURES).
    """
    raise NotImplementedError


def evaluate_and_alert(feat: pd.DataFrame, clf, categories: dict) -> dict:
    """Returns {"report", "alerts", "truth", "scores", "reliability"}.

    report = {"model_test": summary(y_te, p_te, thr, ALERT_BUDGET),
              "rules_test": summary(y_te, rules_score(te), np.inf, ALERT_BUDGET),
              "val_threshold": thr (best_f1_threshold on VAL only), "best_iteration": int}
    alerts (k = report["model_test"]["alerts"] rows, by score desc, stable):
        alert_id 1..k, tx_id, score, score_rank 1..k, score_rank_pct = score_rank/len(te),
        reasons = json.dumps(top_reasons(contrib_row, X_te.iloc[i]), ensure_ascii=False)
        where contrib = clf.booster_.predict(X_te.iloc[top], pred_contrib=True)[:, :-1]
    truth  = te.iloc[top][["tx_id", "is_laundering"(, "pattern")]]  (evaluation only, separate file)
    scores = DataFrame(tx_id, score, is_laundering) over all test rows; reliability = reliability_table(y_te, p_te)
    # TODO(builder): te = feat[split=="test"].reset_index(drop=True); order = np.argsort(-p_te, kind="stable");
    #   alerts frame MUST NOT contain is_laundering or pattern columns.
    """
    raise NotImplementedError


def save_report(out: dict, results_dir: Path) -> None:
    """Write results/model_eval.md (markdown table: model | n | positives | pr_auc | f1_at_val_threshold |
    alerts | precision_at_budget | recall_at_budget; rows "LightGBM", "Rules baseline"; then a line
    "Validation-chosen threshold: {thr:.4f}") and results/reliability_test.csv.
    # TODO(builder): build lines by hand (no tabulate dependency); results_dir.mkdir(parents=True, exist_ok=True).
    """
    raise NotImplementedError

