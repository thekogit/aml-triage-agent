"""Automatic faithfulness check: every evidence item must cite a real tool call, and every number
in the fact must appear (0.5% relative tolerance) in that call's output or arguments."""
from __future__ import annotations

import json
import math
import re
from typing import Any

import numpy as np

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


def extract_source_numbers(entry: ToolLogEntry) -> set[float]:
    """Extract numeric values from JSON values (not raw text) to prevent timestamp/ID fragments from grounding fake numbers."""
    out: set[float] = set()

    def add_val(v: Any) -> None:
        if isinstance(v, (bool, np.bool_)):
            return
        if isinstance(v, (int, float, np.integer, np.floating)):
            try:
                fv = float(v)
                if math.isfinite(fv):
                    out.add(round(fv, 2))
            except (ValueError, OverflowError):
                pass
        elif isinstance(v, str):
            try:
                fv = float(v.replace(",", ""))
                if math.isfinite(fv):
                    out.add(round(fv, 2))
            except ValueError:
                pass
        elif isinstance(v, dict):
            for val in v.values():
                add_val(val)
        elif isinstance(v, (list, tuple)):
            for val in v:
                add_val(val)

    args = entry.get("args")
    if isinstance(args, dict):
        add_val(args)
    elif isinstance(args, str):
        try:
            add_val(json.loads(args))
        except Exception:
            pass

    raw_output = entry.get("output", "")
    if isinstance(raw_output, str):
        try:
            parsed = json.loads(raw_output)
            add_val(parsed)
            if entry.get("name") == "search_policy" and isinstance(parsed, dict):
                for res in parsed.get("results", []):
                    if isinstance(res, dict) and "text" in res:
                        out |= numbers(str(res["text"]))
        except Exception:
            out |= numbers(raw_output)
    elif isinstance(raw_output, (dict, list)):
        add_val(raw_output)

    return out


def grounding_report(assessment: AlertAssessment, tool_log: list[ToolLogEntry]) -> GroundingReport:
    by_id = {e["tool_call_id"]: e for e in tool_log}
    problems: list[GroundingProblem] = []
    for ev in assessment.evidence:
        entry = by_id.get(ev.tool_call_id)
        if entry is None:
            problems.append({"tool_call_id": ev.tool_call_id, "problem": "unknown tool_call_id"})
            continue
        ev_nums = numbers(ev.fact)
        if not ev_nums:
            problems.append({"tool_call_id": ev.tool_call_id, "problem": "fact contains no verifiable numbers"})
            continue
        source = extract_source_numbers(entry)
        missing = sorted(x for x in ev_nums if not contains_number(source, x))
        if missing:
            problems.append({"tool_call_id": ev.tool_call_id, "problem": f"numbers not in tool output: {missing}"})
    n = len(assessment.evidence)
    return {"n_evidence": n, "n_grounded": n - len(problems),
            "grounded_rate": round((n - len(problems)) / n, 3) if n else 0.0, "problems": problems}
