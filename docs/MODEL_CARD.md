# Model Card: AML Detection Model and Triage Assistant

## Intended Use
Rank synthetic bank transactions for anti-money-laundering (AML) review (LightGBM detection model) and prepare structured, cited investigation notes with verifiable facts for human compliance analysts (LangGraph triage assistant).
Demonstration on synthetic data only.

## Not Intended For
- Fully automated closures of alerts without human analyst sign-off.
- Decisions regarding real customers or live production payment networks.
- Regulatory filing (e.g. SAR/STR submission) to financial intelligence units without qualified analyst verification.

## Data
- **Dataset**: IBM AMLworld HI-Small ledger (~5.1 million transactions).
- **Split**: Temporal split by whole calendar days (60% train, 20% validation, 20% test) to prevent future leakage.
- **Customer Profiles**: Synthetic Polish KYC generated via `aml/pii.py` (checksum-valid PESEL and IBAN, declared turnover, risk rating, country).

## Detection Model
- **Algorithm**: LightGBM (Gradient Boosted Decision Trees).
- **Features**: 16 leak-free tabular and graph features computed strictly from prior transactions (24h/7d counts and volumes, pass-through ratio, 2-hop reverse-pair cycles, 7d distinct counterparties).
- **Explaining Alerts**: Per-alert top-5 TreeSHAP contribution reasons computed via LightGBM `pred_contrib=True`.
- **Alert Budget**: Top 0.5% highest scoring transactions in the test split are raised as alerts (`ALERT_BUDGET=0.005`).

### Benchmark Comparison (Test Period)
| Metric | LightGBM Detection | Rules Baseline | Benchmark (Egressy et al. 2024) |
|---|---|---|---|
| PR-AUC | Evaluated on test set | Evaluated on test set | Baseline reference |
| Minority F1 | Evaluated at validation threshold | Evaluated at validation threshold | 0.628 ± 0.003 (with graph GNN) |
| Alert Recall @ 0.5% Budget | Evaluated at budget | Evaluated at budget | N/A |
| Alert Precision @ 0.5% Budget | Evaluated at budget | Evaluated at budget | N/A |

*Full evaluation outputs and calibration curves are persisted in `results/model_eval.md` and `results/reliability_test.csv` upon training.*

## Triage Assistant
- **Framework**: LangGraph `StateGraph` with Tool calling.
- **LLM**: Local `qwen2.5:7b-instruct` (or OpenAI-compatible API).
- **Tools**: 5 deterministic database queries (`get_account_profile`, `get_transactions`, `get_counterparties`, `find_cycles`, `search_policy`).
- **Safety Controls**:
  - PII masking with per-alert stateful redactor.
  - Grounding check matching numbers against parsed JSON values.
  - Fail-safe guard forcing ESCALATE on score rank <= 0.1%, low confidence (< 0.7), high risk, matched typology, or missing transaction checks.
  - Untrusted data tagging protecting tool loop and final decision prompt.

## Known Limitations
- Synthetic IBM ledger data does not reflect all real-world evasion behaviors.
- The policy manual in `aml/policies/` is fictional and designed for system demonstration.
- LLM judge alignment is evaluated on a sample of 30 notes and should not replace compliance quality assurance audits.
- English-only narrative generation.

## Owner & Governance
- Maintained by the engineering team.
- Built following the Project Operations Manual (POM-B).
