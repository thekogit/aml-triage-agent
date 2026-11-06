"""Central configuration. The ONLY module that reads os.environ."""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Skip auto-loading .env during test executions to prevent local settings leaking into tests
if not any("pytest" in arg for arg in sys.argv) and os.getenv("AML_SKIP_DOTENV") != "1" and "PYTEST_CURRENT_TEST" not in os.environ:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:  # pragma: no cover - python-dotenv absent in CI
        pass

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW_TRANS = DATA / "raw" / "HI-Small_Trans.csv"
RAW_PATTERNS = DATA / "raw" / "HI-Small_Patterns.txt"
TX_PATH = DATA / "transactions.parquet"
FEAT_PATH = DATA / "features.parquet"
SCORES_PATH = DATA / "scores_test.parquet"
ALERTS_PATH = DATA / "alerts.parquet"
TRUTH_PATH = DATA / "alert_truth.parquet"
MODEL_PATH = ROOT / "models" / "lgbm.txt"
CATEGORIES_PATH = ROOT / "models" / "categories.json"
RESULTS = ROOT / "results"
POLICY_DIR = Path(__file__).resolve().parent / "policies"
HUMAN_LABELS = ROOT / "labels" / "human_review.csv"

USD_NAME = os.getenv("USD_NAME", "US Dollar")
ALERT_BUDGET = float(os.getenv("ALERT_BUDGET", "0.005"))
HARD_ESCALATE_TOP = float(os.getenv("HARD_ESCALATE_TOP", "0.001"))
MAX_TOOL_CALLS = int(os.getenv("MAX_TOOL_CALLS", "8"))

PG_DSN = os.getenv("PG_DSN", "postgresql://aml:aml@localhost:5432/aml")                         # owner: load, review writes
PG_AGENT_DSN = os.getenv("PG_AGENT_DSN", "postgresql://aml_agent:aml_agent@localhost:5432/aml")  # SELECT-only role
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBED_DIM = 384

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "ollama")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b-instruct")
PRICE_IN_PER_MTOK = float(os.getenv("PRICE_IN_PER_MTOK", "0"))
PRICE_OUT_PER_MTOK = float(os.getenv("PRICE_OUT_PER_MTOK", "0"))

API_URL = os.getenv("API_URL", "http://localhost:8000")
