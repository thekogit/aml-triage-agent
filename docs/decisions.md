# Architectural Decisions & Operations Log

## Record of Architecture Decisions (ADR)

### ADR-001: Detection Alert Budget and Temporal Threshold Tuning
- **Context**: Transaction monitoring models in tier-1 financial institutions operate under strict investigator staffing limits. An unbounded score threshold results in unmanageable alert queues, while purely threshold-based alerting cannot guarantee stable staffing capacity.
- **Decision**:
  - We fix an explicit alert budget of 0.5% (`ALERT_BUDGET = 0.005`) over the test transactions.
  - The model decision threshold is tuned exclusively on validation days (`best_f1_threshold`), ensuring zero future leakage into test scores.
  - Both LightGBM and the red-flag rules baseline are evaluated on the exact same temporal test period.
- **Consequences**:
  - Predictable alert volume matching investigative team bandwidth.
  - Transparent precision/recall trade-offs at fixed operational capacity.

### ADR-002: LLM Selection, Tool Calling Context, and Fail-Safe Guard
- **Context**: The triage agent needs reliable structured function calling, strict JSON output generation, and robust handling of long evidence logs (up to 8 tool responses).
- **Decision**:
  - Primary local model: `qwen2.5:7b-instruct` with 16k context window (`num_ctx 16384`) and temperature 0.0.
  - Fallback/API compatible: OpenAI-compatible endpoint interface.
  - Final decision formatting is handled via a dedicated `finalize` step with JSON schema validation and retry loop.
  - Deterministic guard enforcement: Any `CLOSE` recommendation is overridden to `ESCALATE` if the score rank is within top 0.1%, evidence is ungrounded, confidence < 0.7, a known typology was identified, or transactions were not inspected.
- **Consequences**:
  - Agent cannot bypass compliance controls.
  - High fidelity tool calling without cloud data leakage.

### ADR-003: LLM Judge Calibration & Human Review Agreement
- **Context**: Evaluating LLM-generated investigation notes requires qualitative assessment of faithfulness and usefulness without relying blindly on an unchecked LLM judge.
- **Decision**:
  - A human analyst sample of 30 notes is scored on a 1–5 rubric for faithfulness (factual grounding) and usefulness (actionability).
  - Agreement is calculated using exact match, within-1 tolerance, quadratic weighted Cohen's Kappa, and Spearman rank correlation.
  - The LLM judge evaluates the agent's pre-guard reasoning (`llm_assessment`) so that deterministic guard escalations do not artificially penalize narrative coherence.
- **Consequences**:
  - Calibrated, statistically sound evaluation metrics that distinguish model quality from guard intervention.

### ADR-004: Stateful Per-Alert PII Redactor
- **Context**: In multi-turn investigations involving multiple customer profiles (e.g., sender and receiver), stateless redactors restarting numbering on each tool call mapped different individuals to `<NAME_1>`, risking erroneous conclusions that accounts were owned by the same person.
- **Decision**:
  - `Redactor` retains instance-level name-to-placeholder mappings across all tool calls for a single alert.
  - `triage` explicitly resets the redactor between alerts (`redactor.reset()`).
- **Consequences**:
  - Clean separation of customer identities across investigative tool calls.

### ADR-005: Intra-Minute Temporal Order in Sliding Window Features
- **Context**: Minute-level ledger timestamps with burst fan-outs (e.g. 7 transfers within one minute) allowed same-minute transactions to observe later transactions, causing feature leakage in offline models.
- **Decision**:
  - `window_count_sum` incorporates transaction IDs (`tx_id`) alongside timestamps, ordering events lexicographically by `(group, time, tx_id)`.
  - At query time `t`, only transactions with `time < t` or `(time == t and tx_id <= q_id)` are visible.
- **Consequences**:
  - Strict absence of intra-minute leakage. Offline training feature distributions match real-time scoring.

### ADR-006: Audit Trail Integrity via Exception-Raising Triggers
- **Context**: SQLite-style or PostgreSQL `DO INSTEAD NOTHING` rules silently drop updates and deletes without signaling an error, which can mask bugs or attempted unauthorized tampering.
- **Decision**:
  - Replaced `DO INSTEAD NOTHING` rules with PL/pgSQL triggers (`prevent_audit_modification`) that raise explicit runtime exceptions on any `UPDATE` or `DELETE` against `assessments` or `decisions`.
- **Consequences**:
  - Hard guarantee of append-only audit trail with immediate failure feedback.
