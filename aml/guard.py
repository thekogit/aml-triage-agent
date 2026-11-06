"""Deterministic controls applied after the LLM. The agent may only reduce analyst workload when its
evidence is solid; every doubt resolves to ESCALATE (fail-safe). The guard never downgrades ESCALATE."""
from __future__ import annotations

import math
import re
from typing import Optional

from .schema import AlertAssessment, GroundingReport

CLOSE_MIN_CONFIDENCE = 0.7
_TAG_RE = re.compile(r"<\s*(/?)\s*tool_output", re.IGNORECASE)
_NAME_RE = re.compile(r"[^A-Za-z0-9_]")


def apply_guard(a: AlertAssessment, score_rank_pct: Optional[float], grounding: GroundingReport,
                hard_top: float) -> tuple[AlertAssessment, list[str]]:
    """Return (possibly flipped assessment, reasons). CLOSE -> ESCALATE if ANY rule fires:
    rank in top `hard_top` or rank unknown/NaN; grounded_rate < 1.0; confidence < 0.7;
    risk_level HIGH; typology != NONE (a matched typology cannot be closed by the agent)."""
    reasons: list[str] = []
    if a.recommendation == "CLOSE":
        if score_rank_pct is None or not math.isfinite(score_rank_pct) or score_rank_pct <= hard_top:
            reasons.append(f"model score in the top {hard_top:.1%} or rank unknown: cannot be closed by the agent")
        if float(grounding.get("grounded_rate", 0.0)) < 1.0:
            reasons.append("evidence not fully grounded in tool outputs")
        if a.confidence < CLOSE_MIN_CONFIDENCE:
            reasons.append(f"confidence below {CLOSE_MIN_CONFIDENCE}")
        if a.risk_level == "HIGH":
            reasons.append("risk level HIGH contradicts CLOSE")
        if a.typology != "NONE":
            reasons.append(f"typology {a.typology} matched: closing requires typology NONE")
    if reasons:
        a = a.model_copy(update={"recommendation": "ESCALATE"})
    return a, reasons


def wrap_untrusted(tool_name: str, payload: str) -> str:
    """Tool output goes back to the model as clearly-marked DATA. Any opening/closing tool_output tag
    inside the payload (any case/spacing) is neutralised so the payload cannot escape its wrapper."""
    safe = payload.replace("</tool_output>", "&lt;/tool_output&gt;")
    safe = _TAG_RE.sub(lambda m: f"&lt;{m.group(1)}tool_output", safe)
    name = _NAME_RE.sub("_", tool_name)[:64]
    return f'<tool_output tool="{name}" trust="untrusted-data">\n{safe}\n</tool_output>'
