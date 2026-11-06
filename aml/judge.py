"""LLM-as-judge for analyst notes, validated against 30 human scores."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

try:
    from scipy.stats import spearmanr
except ImportError:  # pragma: no cover
    spearmanr = None

from .schema import JsonLLM, JudgeScore, TriageRun

JUDGE_PROMPT = """You review an AML analyst note against the evidence it cites.
Score each criterion from 1 (poor) to 5 (excellent):
- faithfulness: every statement is supported by the evidence log; no invented numbers or facts.
- usefulness: a second analyst could act on it (typology, key numbers, decision, policy rule).
Return ONLY JSON: {"faithfulness": 1-5, "usefulness": 1-5, "comment": "one sentence"}"""


def judge_run(llm: JsonLLM, run: TriageRun) -> JudgeScore:
    evidence = "\n".join(f"[{e['tool_call_id']}] {e['name']}: {str(e['output'])[:1500]}" for e in run.get("tool_log", []))
    # Score the model's own drafted note so guard intervention does not cause artificial incoherence markdowns
    note_to_score = run.get("llm_assessment", run.get("final_assessment", {}))
    guard_reasons = run.get("guard_reasons", [])
    guard_context = f"\nGuard reasons: {guard_reasons}" if guard_reasons else ""
    note = json.dumps(note_to_score, ensure_ascii=False)
    content, _, _ = llm.complete_json([
        {"role": "system", "content": JUDGE_PROMPT},
        {"role": "user", "content": f"EVIDENCE LOG:\n{evidence}\n\nANALYST NOTE:\n{note}{guard_context}"},
    ])
    try:
        return JudgeScore.model_validate_json(content or "{}")
    except Exception:
        return JudgeScore(faithfulness=1, usefulness=1, comment="Judge response unparseable")


def judge_all(llm: JsonLLM, runs_path: Path, out_path: Path) -> pd.DataFrame:
    rows = []
    lines = [line for line in runs_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for i, line in enumerate(lines):
        try:
            run = json.loads(line)
            alert_id = run.get("alert_id", i + 1)
            score = judge_run(llm, run)
            rows.append({"alert_id": alert_id, **score.model_dump()})
        except Exception as e:
            rows.append({"alert_id": i + 1, "faithfulness": 1, "usefulness": 1, "comment": f"Judge error: {e}"})
        # Incrementally persist after each item to prevent losing progress
        df = pd.DataFrame(rows, columns=["alert_id", "faithfulness", "usefulness", "comment"])
        df.to_csv(out_path, index=False)
    return pd.DataFrame(rows, columns=["alert_id", "faithfulness", "usefulness", "comment"])


def agreement(judge: pd.DataFrame, human: pd.DataFrame) -> dict:
    m = judge.merge(human, on="alert_id", suffixes=("_judge", "_human"))
    out: dict = {"n": int(len(m))}
    nan = float("nan")
    for c in ["faithfulness", "usefulness"]:
        j_col = m[f"{c}_judge"]
        h_col = m[f"{c}_human"]
        diff = (j_col - h_col).abs()
        out[f"{c}_exact"] = round(float((diff == 0).mean()), 3) if len(m) else nan
        out[f"{c}_within_1"] = round(float((diff <= 1).mean()), 3) if len(m) else nan

        # Weighted Cohen's Kappa (quadratic)
        try:
            out[f"{c}_weighted_kappa"] = round(float(cohen_kappa_score(j_col, h_col, weights="quadratic")), 3)
        except Exception:
            out[f"{c}_weighted_kappa"] = nan

        # Spearman rank correlation
        try:
            if spearmanr is not None and len(m) > 1 and j_col.nunique() > 1 and h_col.nunique() > 1:
                res = spearmanr(j_col, h_col)
                out[f"{c}_spearman"] = round(float(res.statistic), 3)
            else:
                out[f"{c}_spearman"] = nan
        except Exception:
            out[f"{c}_spearman"] = nan

    return out
