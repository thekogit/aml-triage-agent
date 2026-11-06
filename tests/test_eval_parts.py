from pathlib import Path

import pandas as pd

from aml.evaluate_agent import agent_metrics, sample_alerts
from aml.judge import agreement
from aml.rag import chunk_policy

POLICY_DIR = Path(__file__).resolve().parent.parent / "aml" / "policies"


def test_sample_alerts_is_balanced_and_reproducible():
    truth = pd.DataFrame({"alert_id": range(1, 101), "is_laundering": [1] * 30 + [0] * 70})
    a = sample_alerts(truth, 40, seed=1)
    assert a == sample_alerts(truth, 40, seed=1)
    picked = truth[truth["alert_id"].isin(a)]
    assert len(a) == 40 and picked["is_laundering"].sum() == 20


def test_agent_metrics():
    runs = pd.DataFrame({
        "ok": [True, True, True, True, False],
        "is_laundering": [1, 1, 0, 0, 1],
        "pattern": ["CYCLE", "FAN-OUT", "NONE", "NONE", "CYCLE"],
        "recommendation": ["ESCALATE", "CLOSE", "ESCALATE", "CLOSE", None],
        "typology": ["CYCLE", "NONE", "FAN-IN", "NONE", None],
        "grounded_rate": [1.0, 1.0, 0.5, 1.0, None],
        "guard_applied": [False, False, True, False, False],
        "tool_calls": [5, 4, 6, 3, 0],
        "latency_s": [10.0, 12.0, 14.0, 8.0, None],
        "cost_usd": [0.002, 0.002, 0.002, 0.002, 0.0],
    })
    m = agent_metrics(runs)
    assert m["invalid_or_failed"] == 1
    assert m["escalate_precision"] == 0.5 and m["escalate_recall"] == 0.5
    assert m["missed_laundering_closed"] == 1
    assert m["typology_accuracy"] == 0.5
    assert m["usd_per_1000_alerts"] == 2.0


def test_judge_agreement():
    j = pd.DataFrame({"alert_id": [1, 2, 3], "faithfulness": [5, 3, 4], "usefulness": [4, 4, 2]})
    h = pd.DataFrame({"alert_id": [1, 2, 3], "faithfulness": [5, 5, 4], "usefulness": [4, 3, 2]})
    a = agreement(j, h)
    assert a["n"] == 3 and a["faithfulness_exact"] == round(2 / 3, 3) and a["usefulness_within_1"] == 1.0


def test_every_policy_chunks_with_doc_id():
    files = sorted(POLICY_DIR.glob("*.md"))
    assert len(files) == 6
    for f in files:
        chunks = chunk_policy(f)
        assert len(chunks) >= 3
        assert chunks[0]["doc_id"].startswith("POL-")


class MockEvalStore:
    def __init__(self, truth_df: pd.DataFrame):
        self._truth = truth_df
        self.memos: dict[int, str | None] = {}

    def truth(self) -> pd.DataFrame:
        return self._truth

    def set_memo(self, tx_id: int, memo: str | None) -> None:
        self.memos[tx_id] = memo


class MockRepo:
    def __init__(self, alerts_map: dict[int, dict]):
        self._alerts = alerts_map

    def alert(self, alert_id: int) -> dict | None:
        return self._alerts.get(alert_id)


def test_run_eval_and_injection_flow(tmp_path):
    from aml.evaluate_agent import injection_test, run_eval
    from aml.pii import Redactor

    truth_df = pd.DataFrame({
        "alert_id": [1, 2],
        "is_laundering": [1, 0],
        "pattern": ["FAN-OUT", "NONE"],
    })
    store = MockEvalStore(truth_df)
    repo = MockRepo({
        1: {"tx_id": 101, "memo": "Initial clean memo"},
        2: {"tx_id": 102, "memo": None},
    })
    redactor = Redactor()

    def mock_triage(r, red, alert_id: int):
        if alert_id == 2:
            raise RuntimeError("Simulated API failure on alert 2")
        # Check if memo on transaction was injected
        tx_memo = store.memos.get(101, "Initial clean memo")
        rec = "CLOSE" if "IGNORE" in (tx_memo or "") else "ESCALATE"
        return {
            "alert_id": alert_id,
            "model": "mock",
            "llm_assessment": {"recommendation": rec},
            "final_assessment": {"recommendation": rec, "typology": "FAN-OUT"},
            "guard_reasons": [],
            "grounding": {"grounded_rate": 1.0},
            "tool_log": [],
            "tool_calls": 2,
            "latency_s": 1.5,
            "cost_usd": 0.001,
        }

    # Test run_eval
    out_dir = tmp_path / "eval"
    metrics = run_eval(repo, store, redactor, n=2, out_dir=out_dir, triage_fn=mock_triage)
    assert metrics["alerts"] == 2
    assert metrics["invalid_or_failed"] == 1
    assert (out_dir / "agent_runs.jsonl").exists()
    assert (out_dir / "agent_runs_summary.csv").exists()
    assert (out_dir / "agent_eval.md").exists()

    # Test injection_test
    inj_df = injection_test(repo, store, redactor, n=1, out_dir=out_dir, triage_fn=mock_triage)
    assert len(inj_df) == 1
    # Check that original memo was restored after test
    assert store.memos[101] == "Initial clean memo"
    assert inj_df.loc[0, "llm_clean_1"] == "ESCALATE"
    assert inj_df.loc[0, "llm_attacked"] == "CLOSE"
    assert bool(inj_df.loc[0, "llm_attack_flipped"]) is True
    assert (out_dir / "injection_test.csv").exists()
