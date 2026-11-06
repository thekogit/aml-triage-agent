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
