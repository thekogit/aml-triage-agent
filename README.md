# AML Triage Agent

Anti-money-laundering (AML) alert detection and triage system powered by a LightGBM scoring model and a LangGraph investigative agent with deterministic safety controls, built on IBM's synthetic bank ledger (AMLworld).

> **Important Notice:** This repository is an engineering research and architecture demonstration running on synthetic data. It is not an active compliance system, does not file Suspicious Activity Reports (SAR/STR) to any financial intelligence unit (FIU/GIIF), and does not constitute legal or regulatory advice. The AML policy manual in `aml/policies/` is fictional.

---

## Architecture Overview

```text
  Kaggle: HI-Small_Trans.csv, HI-Small_Patterns.txt (synthetic)
             │
             ▼
  aml/ingest.py ──► data/transactions.parquet      (+ temporal split, + pattern label)
             │
             ▼
  aml/features.py + aml/windows.py ──► data/features.parquet  (leakage-free window features)
             │
             ▼
  aml/model.py (LightGBM) ──► results/model_eval.md, data/alerts.parquet (+ SHAP reasons)
             │                 data/alert_truth.parquet (evaluation only, isolated from tools)
             ▼
  aml/loader.py ──► Postgres 16 + pgvector (Docker)
                      transactions (serving subset) · accounts (synthetic Polish KYC) · alerts
                      alert_truth (isolated) · policy_chunks (pgvector cosine search)
                      assessments · decisions (append-only audit trail)
             │
  Analyst ──► aml/ui.py (Streamlit :8501) ──► aml/api.py (FastAPI :8000) ──► aml/agent.py triage()
                                                     │
      ┌──────────────────────────────────────────────┴─────────────────────────────────┐
      │ LangGraph:  START → agent ⇄ tools → END                                         │
      │   tools = ToolBox: run investigate.py function → redact PII → wrap as untrusted │
      │           → log {tool_call_id, args, output} (budget: MAX_TOOL_CALLS)          │
      │ finalize: strict JSON (AlertAssessment), validated with bounded retries        │
      │ grounding: numbers verified against tool output JSON values (not raw text)     │
      │ guard: fail-safe escalation (rank in top 0.1%, low confidence, missing tool)  │
      └──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Key Safety Controls & Defenses

1. **Deterministic Guard**: The agent recommends, but a deterministic rule-based guard enforces fail-safe escalation. Any alert with a score rank in the top 0.1%, ungrounded evidence, confidence below 0.7, high risk rating, active typology, or missing transaction checks is forced to `ESCALATE`.
2. **Grounding & Verifiable Evidence**: Every fact cited by the LLM must reference a specific `tool_call_id`. Numbers are matched against parsed JSON values rather than raw text (preventing timestamp fragments like `12:45` from validating hallucinated figures).
3. **Prompt Injection Containment**: All tool outputs (including transaction memos) are sanitized and wrapped in `<tool_output trust="untrusted-data">` blocks. This wrapping protects both the agentic loop and the final decision prompt (`finalize`).
4. **PII Masking**: Customer names, valid Polish PESELs, and IBANs are redacted into stable placeholders (`<NAME_1>`, `<IBAN_1>`) with per-alert redactor state to avoid cross-profile identity collisions. PESEL is never loaded into agent-accessible views.
5. **Least-Privilege Database Role**: The agent queries PostgreSQL via `aml_agent` (SELECT-only grant on transactions, accounts without PESEL, alerts, and policies). `alert_truth` is strictly isolated. Audit trail tables (`assessments`, `decisions`) are append-only with exception-raising triggers preventing UPDATE or DELETE.
6. **Leakage-Free Feature Engineering**: Past-only sliding window aggregations in O(n log n) using numpy. Within the same minute timestamp, transactions only see prior events (`tx_id <= query_id`), eliminating intra-minute leakage in burst scenarios.

---

## Quickstart

### Prerequisites
- Python 3.12
- Docker Desktop (for Postgres 16 + pgvector)
- Ollama or an OpenAI-compatible endpoint (optional, for LLM triage)

### Setup Virtual Environment
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements-test.txt -r requirements-dev.txt
pytest -q
ruff check aml tests
```

### End-to-End Pipeline
```bash
# 1. Download Kaggle HI-Small CSVs into data/raw/
python -m aml.cli check-raw

# 2. Ingest transactions and assign temporal splits (train/val/test)
python -m aml.cli ingest

# 3. Compute leak-free window and graph features
python -m aml.cli features

# 4. Train LightGBM model and compare against rules baseline
python -m aml.cli train

# 5. Start serving database and load data
docker compose up -d db
python -m aml.cli load-db
python -m aml.cli index-policies
python -m aml.cli db-check

# 6. Run agent triage and evaluations
python -m aml.cli triage --alert-id 1
python -m aml.cli eval-agent --n 200
python -m aml.cli injection-test --n 10
python -m aml.cli judge

# 7. Start API and Review UI
docker compose up --build
```

---

## Evaluation Methodology

The system evaluates:
- **Detection Model**: PR-AUC, minority class F1, precision and recall at fixed alert budget (0.5%) on temporal test split vs. rules baseline.
- **Agent Decisions**: Escalate precision/recall, closed share (both sample 50/50 and population-prevalence weighted), missed laundering closed, typology accuracy, grounded rate, and tool call latency.
- **Adversarial Resilience**: Prompt injection test measuring both model flip rate (with clean-vs-clean baseline) and post-guard protection.
- **Judge Agreement**: LLM-as-judge scored against human analyst reviews using exact match, within-1 tolerance, quadratic weighted Cohen's Kappa, and Spearman rank correlation.

---

## Repository Structure

```text
├── aml/
│   ├── config.py         # Central settings and environment variables
│   ├── db.py             # Postgres schema, triggers, PgRepo & PgStore
│   ├── ingest.py         # Raw transaction/pattern parsing & temporal splits
│   ├── windows.py        # O(n log n) leak-free window kernels
│   ├── features.py       # Graph and time-window feature pipeline
│   ├── model.py          # LightGBM detection model & rules baseline
│   ├── pii.py            # Polish PESEL/IBAN generators & stateful redactor
│   ├── investigate.py    # Core investigation tools (accounts, txs, cycles, policy)
│   ├── schema.py         # Pydantic v2 data contracts & protocols
│   ├── grounding.py      # Numerical evidence verification against JSON values
│   ├── guard.py          # Deterministic fail-safe guard & untrusted wrapper
│   ├── toolbox.py        # Safe tool execution with MAX_TOOL_CALLS clamping
│   ├── finalize.py       # Strict JSON synthesis, retry loop & postprocessing
│   ├── agent.py          # LangGraph state machine & triage workflow
│   ├── loader.py         # Serving database loader for transactions & KYC
│   ├── rag.py            # Policy manual chunking & pgvector indexing
│   ├── evaluate_agent.py # Agent metrics & prompt injection test
│   ├── judge.py          # LLM judge & human agreement (Kappa/Spearman)
│   ├── api.py            # FastAPI service (:8000)
│   ├── ui.py             # Streamlit review UI (:8501)
│   └── cli.py            # Unified command-line interface
├── aml/policies/         # Fictional bank AML policy documents (POL-01..06)
├── tests/                # Unit, contract, and safety test suite
├── docs/                 # MODEL_CARD.md, RISK_NOTES.md, decisions.md
├── results/              # Evaluation outputs (model_eval.md, agent_eval.md, etc.)
└── labels/               # Human review labels for judge calibration
```

---

## License

MIT License. Synthetic data source: IBM Transactions for Anti-Money Laundering (AMLworld) under Community Data License Agreement (CDLA).
