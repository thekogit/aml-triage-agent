"""Synthetic Polish KYC (checksum-valid PESEL and IBAN) and a deterministic PII redactor.

The Kaggle data has no customers, so every account gets a synthetic profile. The redactor
removes PESEL, Polish IBANs and known names before any text reaches an LLM.
"""
from __future__ import annotations

import re
from datetime import date, timedelta
from typing import Iterable

import numpy as np
import pandas as pd

PESEL_WEIGHTS = [1, 3, 7, 9, 1, 3, 7, 9, 1, 3]
FIRST_M = ["Jan", "Piotr", "Krzysztof", "Andrzej", "Tomasz", "Paweł", "Michał", "Marcin", "Jakub", "Adam"]
FIRST_F = ["Anna", "Maria", "Katarzyna", "Małgorzata", "Agnieszka", "Barbara", "Ewa", "Magdalena", "Joanna", "Zofia"]
LAST = ["Nowak", "Kowalski", "Wiśniewski", "Wójcik", "Kowalczyk", "Kamiński", "Lewandowski", "Zieliński",
        "Szymański", "Woźniak", "Dąbrowski", "Kozłowski", "Jankowski", "Mazur", "Kwiatkowski"]
COMPANY_SUFFIX = ["sp. z o.o.", "S.A.", "sp. k.", "sp. j."]
COMPANY_WORDS = ["Trans", "Bud", "Pol", "Tech", "Agro", "Med", "Logistics", "Trade", "Invest", "Serwis"]
PROFILE_COLUMNS = ["account", "customer_type", "name", "pesel", "iban", "risk_rating",
                   "declared_monthly_turnover_usd", "onboarding_date", "country"]


# ---------- PESEL ----------
def pesel_check_digit(first10: str) -> int:
    return (10 - sum(int(d) * w for d, w in zip(first10, PESEL_WEIGHTS)) % 10) % 10


def pesel_is_valid(pesel: str) -> bool:
    return bool(re.fullmatch(r"\d{11}", pesel)) and pesel_check_digit(pesel[:10]) == int(pesel[10])


def make_pesel(birth: date, female: bool, serial: int) -> str:
    year = birth.year
    month = birth.month + {18: 80, 19: 0, 20: 20, 21: 40, 22: 60}[year // 100]
    body = f"{year % 100:02d}{month:02d}{birth.day:02d}{serial % 1000:03d}"
    sex_digit = (serial % 5) * 2 + (0 if female else 1)  # even = female, odd = male
    first10 = body + str(sex_digit)
    return first10 + str(pesel_check_digit(first10))


# ---------- IBAN (Poland: PL + 2 check digits + 24 digits) ----------
def _iban_numeric(s: str) -> int:
    return int("".join(str(int(ch, 36)) for ch in s))


def make_pl_iban(bban24: str) -> str:
    if not re.fullmatch(r"\d{24}", bban24):
        raise ValueError("Polish BBAN must be 24 digits")
    check = 98 - _iban_numeric(bban24 + "PL00") % 97
    return f"PL{check:02d}{bban24}"


def iban_is_valid(iban: str) -> bool:
    s = iban.replace(" ", "").upper()
    if not re.fullmatch(r"[A-Z]{2}\d{2}[A-Z0-9]{10,30}", s):
        return False
    return _iban_numeric(s[4:] + s[:4]) % 97 == 1


# ---------- synthetic KYC ----------
def synthetic_profiles(accounts: Iterable[str], seed: int = 7) -> pd.DataFrame:
    """One profile per distinct account, deterministic for (set(accounts), seed): input order is irrelevant."""
    rng = np.random.default_rng(seed)
    rows = []
    for i, acc in enumerate(sorted(set(accounts))):
        kind = rng.choice(["INDIVIDUAL", "SOLE_TRADER", "COMPANY"], p=[0.55, 0.2, 0.25])
        female = bool(rng.random() < 0.5)
        first = rng.choice(FIRST_F if female else FIRST_M)
        last = rng.choice(LAST)
        if female and last.endswith("ski"):
            last = last[:-1] + "a"   # Kowalski -> Kowalska
        birth = date(1950, 1, 1) + timedelta(days=int(rng.integers(0, 365 * 55)))
        if kind == "COMPANY":
            name = f"{rng.choice(COMPANY_WORDS)}{rng.choice(COMPANY_WORDS).lower()} {rng.choice(COMPANY_SUFFIX)}"
            pesel = None
        else:
            name = f"{first} {last}" + (f" {rng.choice(COMPANY_WORDS)}" if kind == "SOLE_TRADER" else "")
            pesel = make_pesel(birth, female, i)
        rows.append({
            "account": acc,
            "customer_type": str(kind),
            "name": str(name),
            "pesel": pesel,
            "iban": make_pl_iban("".join(str(d) for d in rng.integers(0, 10, 24))),
            "risk_rating": str(rng.choice(["LOW", "MEDIUM", "HIGH"], p=[0.7, 0.25, 0.05])),
            "declared_monthly_turnover_usd": float(round(rng.lognormal(9 if kind != "COMPANY" else 12, 1), -2)),
            "onboarding_date": date(2015, 1, 1) + timedelta(days=int(rng.integers(0, 365 * 7))),
            "country": "PL" if rng.random() < 0.9 else str(rng.choice(["DE", "CZ", "UA", "LT", "GB"])),
        })
    return pd.DataFrame(rows, columns=PROFILE_COLUMNS)


# ---------- redaction ----------
# 11 digits not embedded in a longer number or a decimal amount (e.g. "44051401359.5" is an amount).
PESEL_RE = re.compile(r"(?<!\d)(?<!\d[.,])\d{11}(?!\d)(?![.,]\d)")
IBAN_RE = re.compile(r"\bPL\s?\d{2}(?:\s?\d{4}){6}\b")


class Redactor:
    """Replace checksum-valid PESEL, checksum-valid Polish IBANs and known customer names with
    placeholders <KIND_n>. Numbering is stable across redact() calls for the same Redactor instance.
    Call reset() to clear mappings between alerts.
    Invalid-checksum numbers are left untouched so amounts/ids survive for grounding.
    Input must be ensure_ascii=False JSON."""

    def __init__(self, known_names: Iterable[str] = ()):
        names = sorted({n for n in known_names if isinstance(n, str) and len(n) >= 3}, key=len, reverse=True)
        self._name_re = (re.compile(r"(?<!\w)(?:" + "|".join(re.escape(n) for n in names) + r")(?!\w)")
                         if names else None)
        self.mapping: dict[str, str] = {}
        self.reverse: dict[str, str] = {}

    def reset(self) -> None:
        self.mapping.clear()
        self.reverse.clear()

    def redact(self, text: str, mapping: dict[str, str] | None = None) -> tuple[str, dict[str, str]]:
        target_mapping = self.mapping if mapping is None else mapping
        reverse = self.reverse if mapping is None else {v: k for k, v in target_mapping.items()}

        def sub(kind: str, value: str) -> str:
            if value in reverse:
                return reverse[value]
            key = f"<{kind}_{sum(1 for k in target_mapping if k.startswith('<' + kind + '_')) + 1}>"
            target_mapping[key] = value
            reverse[value] = key
            return key

        text = IBAN_RE.sub(lambda m: sub("IBAN", m.group(0)) if iban_is_valid(m.group(0)) else m.group(0), text)
        text = PESEL_RE.sub(lambda m: sub("PESEL", m.group(0)) if pesel_is_valid(m.group(0)) else m.group(0), text)
        if self._name_re:
            text = self._name_re.sub(lambda m: sub("NAME", m.group(0)), text)
        return text, dict(target_mapping)
