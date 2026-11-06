"""Cross-module data contracts: LLM output models, run/audit records, and storage Protocols.

Every boundary in the system is typed here. No module defines its own ad-hoc dict shape.
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional, Protocol, TypedDict, runtime_checkable

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Typology = Literal["FAN-OUT", "FAN-IN", "GATHER-SCATTER", "SCATTER-GATHER", "CYCLE", "RANDOM",
                   "BIPARTITE", "STACK", "NONE", "OTHER"]
Recommendation = Literal["ESCALATE", "CLOSE"]
RiskLevel = Literal["LOW", "MEDIUM", "HIGH"]
DecisionAction = Literal["APPROVE", "OVERRIDE_ESCALATE", "OVERRIDE_CLOSE"]
AlertStatus = Literal["OPEN", "TRIAGED", "DECIDED"]

KNOWN_PATTERNS: frozenset[str] = frozenset(
    {"FAN-OUT", "FAN-IN", "GATHER-SCATTER", "SCATTER-GATHER", "CYCLE", "RANDOM", "BIPARTITE", "STACK"})
TOOL_NAMES: tuple[str, ...] = ("get_account_profile", "get_transactions", "get_counterparties",
                               "find_cycles", "search_policy")
# Columns of every transaction DataFrame returned by Repo.transactions (serving schema).
TX_COLUMNS: list[str] = ["tx_id", "ts", "src", "dst", "amount_paid", "payment_currency", "amount_received",
                         "receiving_currency", "payment_format", "amount_usd", "memo"]
# Label-bearing names. They may appear ONLY in offline code, db.py/loader.py and evaluation code.
TRUTH_NAMES: tuple[str, ...] = ("alert_truth", "is_laundering", "pattern")


# ---------------------------------------------------------------- LLM output contract
class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tool_call_id: str = Field(..., min_length=1, description="id of the tool call whose output proves this fact")
    fact: str = Field(..., min_length=5, max_length=300)


class AlertAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    alert_id: int
    recommendation: Recommendation
    risk_level: RiskLevel
    typology: Typology
    evidence: list[Evidence] = Field(..., min_length=1, max_length=8)
    narrative: str = Field(..., min_length=20)
    confidence: float = Field(..., ge=0.0, le=1.0)

    @field_validator("narrative")
    @classmethod
    def at_most_150_words(cls, v: str) -> str:
        if len(v.split()) > 150:
            raise ValueError("narrative must be at most 150 words")
        return v.strip()


JSON_SHAPE = """{
  "alert_id": 123,
  "recommendation": "ESCALATE | CLOSE",
  "risk_level": "LOW | MEDIUM | HIGH",
  "typology": "FAN-OUT | FAN-IN | GATHER-SCATTER | SCATTER-GATHER | CYCLE | RANDOM | BIPARTITE | STACK | NONE | OTHER",
  "evidence": [{"tool_call_id": "call_abc", "fact": "one fact copied from that tool's output, numbers unchanged"}],
  "narrative": "max 150 words for the analyst",
  "confidence": 0.0
}"""


class JudgeScore(BaseModel):
    model_config = ConfigDict(extra="ignore")
    faithfulness: int = Field(..., ge=1, le=5)
    usefulness: int = Field(..., ge=1, le=5)
    comment: str = ""


# ---------------------------------------------------------------- human decision contract
class Decision(BaseModel):
    """Analyst decision. The ONLY path that ends an alert. Overrides require a comment."""
    model_config = ConfigDict(extra="forbid")
    analyst: str = Field(..., min_length=2, max_length=64)
    action: DecisionAction
    comment: str = Field("", max_length=2000)
    assessment_id: Optional[int] = None

    @model_validator(mode="after")
    def override_needs_comment(self) -> "Decision":
        if self.action != "APPROVE" and not self.comment.strip():
            raise ValueError("overrides require a non-empty comment")
        return self


# ---------------------------------------------------------------- records (plain JSON dicts)
class PolicyChunk(TypedDict):
    doc_id: str
    title: str
    text: str


class ToolLogEntry(TypedDict):
    tool_call_id: str
    name: str
    args: dict
    output: str          # redacted JSON text, NOT wrapped
    latency_s: float


class GroundingProblem(TypedDict):
    tool_call_id: str
    problem: str


class GroundingReport(TypedDict):
    n_evidence: int
    n_grounded: int
    grounded_rate: float
    problems: list[GroundingProblem]


class AlertContext(TypedDict):
    alert_id: int
    tx_id: int
    timestamp: str                 # str(pd.Timestamp), ISO-parsable
    from_account: str
    to_account: str
    amount_paid: Optional[float]
    payment_currency: str
    amount_usd: Optional[float]
    payment_format: str
    model_score: float
    score_rank: int
    score_rank_pct: float
    model_reasons: str             # JSON text: list of {feature, value, contribution}


class TriageRun(TypedDict):
    alert_id: int
    model: str
    llm_assessment: dict           # AlertAssessment.model_dump() BEFORE guard
    final_assessment: dict         # AlertAssessment.model_dump() AFTER guard
    guard_reasons: list[str]
    grounding: GroundingReport
    tool_log: list[ToolLogEntry]
    tool_calls: int
    latency_s: float
    tokens_in: int
    tokens_out: int
    cost_usd: float


# ---------------------------------------------------------------- Protocols
@runtime_checkable
class Repo(Protocol):
    """READ-ONLY view used by investigation tools and the agent. Never exposes truth or writes."""

    def alert(self, alert_id: int) -> Optional[dict]: ...
    def profile(self, account: str) -> Optional[dict]: ...
    def transactions(self, accounts: list[str], start: datetime, end: datetime, limit: int) -> pd.DataFrame: ...
    def search_policy(self, query: str, k: int) -> list[PolicyChunk]: ...


@runtime_checkable
class ReviewStore(Protocol):
    """Writes allowed by the system. Used by api.py only, never by tools/agent."""

    def list_alerts(self, status: AlertStatus = "OPEN", limit: int = 50) -> list[dict]: ...
    def latest_assessment(self, alert_id: int) -> Optional[dict]: ...
    def save_assessment(self, run: TriageRun) -> int: ...
    def save_decision(self, alert_id: int, decision: Decision) -> int: ...


@runtime_checkable
class EvalStore(Protocol):
    """Evaluation-only access to ground truth and memo injection. Never passed to the agent."""

    def truth(self) -> pd.DataFrame: ...
    def set_memo(self, tx_id: int, memo: Optional[str]) -> None: ...


@runtime_checkable
class JsonLLM(Protocol):
    """Minimal JSON-mode chat client. Returns (content, tokens_in, tokens_out)."""

    def complete_json(self, messages: list[dict[str, str]]) -> tuple[str, int, int]: ...
