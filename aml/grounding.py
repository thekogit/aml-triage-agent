"""Automatic faithfulness check: every evidence item must cite a real tool call, and every number
in the fact must appear (0.5% relative tolerance) in that call's output or arguments."""
from __future__ import annotations

import re

from .schema import AlertAssessment, GroundingProblem, GroundingReport, ToolLogEntry

NUM_RE = re.compile(r"\d+(?:[.,]\d+)*")
REL_TOL = 0.005  # never raise this to make grounding pass (ABN-B-10)


def numbers(text: str) -> set[float]:
    """All numbers in text, thousands separators removed, rounded to 2 dp ("1,050.00" -> 1050.0)."""
    out: set[float] = set()
    for tok in NUM_RE.findall(text):
        try:
            out.add(round(float(tok.replace(",", "")), 2))
        except ValueError:
            continue
    return out


def contains_number(haystack: set[float], x: float, rel_tol: float = REL_TOL) -> bool:
    return any(abs(x - h) <= rel_tol * max(abs(h), 1.0) for h in haystack)


def grounding_report(assessment: AlertAssessment, tool_log: list[ToolLogEntry]) -> GroundingReport:
    by_id = {e["tool_call_id"]: e for e in tool_log}
    problems: list[GroundingProblem] = []
    for ev in assessment.evidence:
        entry = by_id.get(ev.tool_call_id)
        if entry is None:
            problems.append({"tool_call_id": ev.tool_call_id, "problem": "unknown tool_call_id"})
            continue
        source = numbers(entry["output"]) | numbers(str(entry.get("args", "")))
        missing = sorted(x for x in numbers(ev.fact) if not contains_number(source, x))
        if missing:
            problems.append({"tool_call_id": ev.tool_call_id, "problem": f"numbers not in tool output: {missing}"})
    n = len(assessment.evidence)
    return {"n_evidence": n, "n_grounded": n - len(problems),
            "grounded_rate": round((n - len(problems)) / n, 3) if n else 0.0, "problems": problems}
