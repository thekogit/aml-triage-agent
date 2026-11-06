"""Tools exposed to the LLM: schemas (stub functions whose names/docstrings/args the model sees) and
the ToolBox executor: validate+clamp args -> run investigate.* -> redact -> log -> wrap as untrusted."""
from __future__ import annotations

import json
import time
from datetime import datetime
from typing import Any

from . import investigate as inv
from .guard import wrap_untrusted
from .pii import Redactor
from .schema import TOOL_NAMES, Repo, ToolLogEntry


# ---- schemas the model sees. Bodies are never executed. ----
def get_account_profile(account: str) -> str:
    """Customer profile (KYC) for one account: customer type, risk rating, declared monthly turnover,
    onboarding date, country. Names and IDs are redacted.

    Args:
        account: account id such as "70_100428660"
    """
    raise NotImplementedError


def get_transactions(account: str, hours_before: int = 72, hours_after: int = 24) -> str:
    """Transactions of one account around the alert time, with totals in and out (USD).

    Args:
        account: account id
        hours_before: hours before the alert to include (1 to 168)
        hours_after: hours after the alert to include (0 to 72)
    """
    raise NotImplementedError


def get_counterparties(account: str, hours: int = 72) -> str:
    """Top payees and payers of one account around the alert time, with counts and USD sums.

    Args:
        account: account id
        hours: window on each side of the alert time (1 to 168)
    """
    raise NotImplementedError


def find_cycles(account: str, hours: int = 168, max_len: int = 4) -> str:
    """Time-ordered payment cycles that leave the account and return to it within the window.

    Args:
        account: account id
        hours: window on each side of the alert time (1 to 336)
        max_len: maximum number of accounts in a cycle (2 to 4)
    """
    raise NotImplementedError


def search_policy(query: str) -> str:
    """Search the bank's internal AML policy manual. Use it to name the typology and the escalation rule.

    Args:
        query: what to look for, e.g. "fan-out structuring escalation threshold"
    """
    raise NotImplementedError


TOOL_SPECS = [get_account_profile, get_transactions, get_counterparties, find_cycles, search_policy]
assert tuple(f.__name__ for f in TOOL_SPECS) == TOOL_NAMES


def _int(a: dict, key: str, default: int, lo: int, hi: int) -> int:
    v = a.get(key, default)
    if isinstance(v, bool) or v is None:
        v = default
    return max(lo, min(int(v), hi))          # non-numeric -> ValueError -> logged as tool error


def _account(a: dict) -> str:
    acc = a.get("account")
    if not isinstance(acc, str) or not acc.strip() or len(acc) > 64:
        raise ValueError("account must be a non-empty string of at most 64 characters")
    return acc.strip()


class ToolBox:
    """Executes tool calls for ONE alert. Holds only a read-only Repo. Never raises."""

    def __init__(self, repo: Repo, redactor: Redactor, alert_ts: datetime):
        self.repo, self.redactor, self.alert_ts = repo, redactor, alert_ts
        self.log: list[ToolLogEntry] = []

    def execute(self, name: str, args: Any, call_id: str) -> str:
        started = time.perf_counter()
        args = args if isinstance(args, dict) else {}
        try:
            result = self._dispatch(name, args)
        except Exception as e:  # a failing tool must not crash the investigation
            result = {"error": f"{type(e).__name__}: {e}"}
        text = json.dumps(result, ensure_ascii=False, default=str)  # ensure_ascii=False: names stay matchable
        text, _ = self.redactor.redact(text)
        self.log.append({"tool_call_id": str(call_id), "name": str(name), "args": args, "output": text,
                         "latency_s": round(time.perf_counter() - started, 3)})
        return wrap_untrusted(str(name), text)

    def _dispatch(self, name: str, a: dict) -> dict:
        t = self.alert_ts
        if name == "get_account_profile":
            return inv.account_profile(self.repo, _account(a), self.redactor)
        if name == "get_transactions":
            return inv.recent_transactions(self.repo, _account(a), t, _int(a, "hours_before", 72, 1, 168),
                                           _int(a, "hours_after", 24, 0, 72))
        if name == "get_counterparties":
            return inv.counterparties(self.repo, _account(a), t, _int(a, "hours", 72, 1, 168))
        if name == "find_cycles":
            return inv.find_cycles(self.repo, _account(a), t, _int(a, "hours", 168, 1, 336),
                                   _int(a, "max_len", 4, 2, 4))
        if name == "search_policy":
            q = a.get("query")
            if not isinstance(q, str) or not q.strip():
                raise ValueError("query must be a non-empty string")
            return inv.search_policy(self.repo, q[:500], 3)
        return {"error": f"unknown tool {name}"}
