"""LLM-as-judge for analyst notes, validated against 30 human scores."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .schema import JsonLLM, JudgeScore, TriageRun

JUDGE_PROMPT = """You review an AML analyst note against the evidence it cites.
Score each criterion from 1 (poor) to 5 (excellent):
- faithfulness: every statement is supported by the evidence log; no invented numbers or facts.
- usefulness: a second analyst could act on it (typology, key numbers, decision, policy rule).
Return ONLY JSON: {"faithfulness": 1-5, "usefulness": 1-5, "comment": "one sentence"}"""


def judge_run(llm: JsonLLM, run: TriageRun) -> JudgeScore:
    evidence = "\n".join(f"[{e['tool_call_id']}] {e['name']}: {e['output'][:1500]}" for e in run["tool_log"])
    note = json.dumps(run["final_assessment"], ensure_ascii=False)
    content, _, _ = llm.complete_json([
        {"role": "system", "content": JUDGE_PROMPT},
        {"role": "user", "content": f"EVIDENCE LOG:\n{evidence}\n\nANALYST NOTE:\n{note}"}])
    return JudgeScore.model_validate_json(content or "{}")


def judge_all(llm: JsonLLM, runs_path: Path, out_path: Path) -> pd.DataFrame:
    rows = []
    for line in runs_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            run = json.loads(line)
            rows.append({"alert_id": run["alert_id"], **judge_run(llm, run).model_dump()})
    df = pd.DataFrame(rows, columns=["alert_id", "faithfulness", "usefulness", "comment"])
    df.to_csv(out_path, index=False)
    return df


def agreement(judge: pd.DataFrame, human: pd.DataFrame) -> dict:
    m = judge.merge(human, on="alert_id", suffixes=("_judge", "_human"))
    out: dict = {"n": int(len(m))}
    for c in ["faithfulness", "usefulness"]:
        diff = (m[f"{c}_judge"] - m[f"{c}_human"]).abs()
        out[f"{c}_exact"] = round(float((diff == 0).mean()), 3) if len(m) else float("nan")
        out[f"{c}_within_1"] = round(float((diff <= 1).mean()), 3) if len(m) else float("nan")
    return out
