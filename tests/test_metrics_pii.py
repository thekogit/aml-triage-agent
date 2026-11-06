from datetime import date

import numpy as np
import pandas as pd

from aml.metrics import at_budget, best_f1_threshold, summary
from aml.model import rules_score
from aml.pii import Redactor, iban_is_valid, make_pesel, make_pl_iban, pesel_is_valid, synthetic_profiles


def test_known_valid_pesel_and_iban():
    assert pesel_is_valid("44051401359")                  # textbook example PESEL
    assert not pesel_is_valid("44051401358")
    assert iban_is_valid("PL61 1090 1014 0000 0712 1981 2874")  # textbook example Polish IBAN
    assert not iban_is_valid("PL61 1090 1014 0000 0712 1981 2875")


def test_generated_ids_are_valid():
    for i in range(50):
        assert pesel_is_valid(make_pesel(date(1990 + i % 20, 1 + i % 12, 1 + i % 28), female=i % 2 == 0, serial=i))
    assert make_pesel(date(2005, 3, 9), female=True, serial=0)[2:4] == "23"   # 2000s: month + 20
    assert iban_is_valid(make_pl_iban("109010140000071219812874"))


def test_profiles_are_deterministic_and_valid():
    a = synthetic_profiles(["1_A", "2_B", "3_C"], seed=1)
    b = synthetic_profiles(["3_C", "1_A", "2_B"], seed=1)
    pd.testing.assert_frame_equal(a, b)
    assert a["iban"].map(iban_is_valid).all()
    assert a["pesel"].dropna().map(pesel_is_valid).all()


def test_redactor_removes_pii_and_keeps_invalid_numbers():
    r = Redactor(["Anna Kowalska"])
    text = "Klientka Anna Kowalska, PESEL 44051401359, konto PL61 1090 1014 0000 0712 1981 2874, nr zlecenia 12345678901."
    out, mapping = r.redact(text)
    assert "Anna Kowalska" not in out and "44051401359" not in out and "PL61" not in out
    assert "<NAME_1>" in out and "<PESEL_1>" in out and "<IBAN_1>" in out
    assert "12345678901" in out          # 11 digits but invalid checksum: not a PESEL, left alone
    assert mapping["<PESEL_1>"] == "44051401359"


def test_redactor_multi_call_shared_mapping():
    r = Redactor(["Jan Kowalski", "Anna Nowak"])
    # First call: Jan Kowalski
    out1, m1 = r.redact("Account 1 belongs to Jan Kowalski.")
    assert "<NAME_1>" in out1
    assert m1["<NAME_1>"] == "Jan Kowalski"

    # Second call: Anna Nowak (must get <NAME_2>, not restart at <NAME_1>)
    out2, m2 = r.redact("Account 2 belongs to Anna Nowak.")
    assert "<NAME_2>" in out2
    assert "<NAME_1>" not in out2
    assert m2["<NAME_2>"] == "Anna Nowak"

    # Third call: Jan Kowalski again (must reuse <NAME_1>)
    out3, m3 = r.redact("Counterparty was Jan Kowalski.")
    assert "<NAME_1>" in out3
    assert "<NAME_2>" not in out3
    assert "<NAME_3>" not in out3

    # Reset clears mappings
    r.reset()
    out4, m4 = r.redact("New alert for Anna Nowak.")
    assert "<NAME_1>" in out4
    assert m4["<NAME_1>"] == "Anna Nowak"


def test_budget_and_threshold_metrics():
    y = np.array([0, 0, 1, 0, 1, 0, 0, 0, 0, 1])
    p = np.array([.1, .2, .9, .3, .8, .1, .2, .3, .1, .4])
    b = at_budget(y, p, 0.2)
    assert b == {"alerts": 2, "precision_at_budget": 1.0, "recall_at_budget": 2 / 3}
    thr = best_f1_threshold(y, p)
    assert 0.3 < thr <= 0.4
    s = summary(y, p, thr, 0.2)
    assert s["f1_at_val_threshold"] > 0.8 and s["pr_auc"] > 0.9


def test_rules_score_counts_flags():
    feat = pd.DataFrame({
        "src_out_cnt_24h": [6, 1], "dst_in_cnt_24h": [6, 1], "src_in_usd_24h": [100.0, 0.0],
        "passthrough_ratio": [0.9, 0.0], "reverse_pair_cnt_7d": [1, 0], "log_amount_usd": [5.0, 5.0],
    })
    s = rules_score(feat)
    assert s[0] > 4 and s[1] < 1
