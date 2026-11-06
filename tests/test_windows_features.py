import numpy as np
import pandas as pd

from aml.features import FEATURES, build_features
from aml.ingest import attach_patterns, read_patterns, temporal_split
from aml.windows import window_count_sum


def brute(ev_g, ev_t, ev_v, q_g, q_t, w):
    cnt, tot = [], []
    for g, t in zip(q_g, q_t):
        m = (ev_g == g) & (ev_t > t - w) & (ev_t <= t)
        cnt.append(m.sum())
        tot.append(ev_v[m].sum())
    return np.array(cnt), np.array(tot)


def test_window_count_sum_matches_brute_force():
    rng = np.random.default_rng(0)
    ev_g = rng.integers(0, 5, 400)
    ev_t = rng.integers(0, 10_000, 400)
    ev_v = rng.random(400)
    q_g = rng.integers(-1, 5, 200)
    q_t = rng.integers(0, 10_000, 200)
    c, s = window_count_sum(ev_g, ev_t, ev_v, q_g, q_t, 1500)
    bc, bs = brute(ev_g, ev_t, ev_v, q_g, q_t, 1500)
    bc[q_g < 0], bs[q_g < 0] = 0, 0.0
    assert np.array_equal(c, bc)
    assert np.allclose(s, bs)


def toy_transactions():
    rows = [
        # ts,               src,   dst,   paid, cur,        recv, rcur,       fmt,    laundering
        ("2022-09-01 00:10", "A", "B", 100.0, "US Dollar", 100.0, "US Dollar", "Wire", 0),
        ("2022-09-01 01:00", "B", "C", 90.0, "US Dollar", 90.0, "US Dollar", "Wire", 1),
        ("2022-09-01 02:00", "C", "A", 80.0, "US Dollar", 80.0, "US Dollar", "Wire", 1),
        ("2022-09-02 03:00", "A", "B", 50.0, "Euro", 55.0, "US Dollar", "ACH", 0),
        ("2022-09-02 04:00", "B", "A", 40.0, "US Dollar", 40.0, "US Dollar", "ACH", 0),
        ("2022-09-03 05:00", "A", "D", 10.0, "US Dollar", 10.0, "US Dollar", "Cash", 0),
    ]
    tx = pd.DataFrame(rows, columns=["ts", "src", "dst", "amount_paid", "payment_currency", "amount_received",
                                     "receiving_currency", "payment_format", "is_laundering"])
    tx["ts"] = pd.to_datetime(tx["ts"])
    tx.insert(0, "tx_id", range(len(tx)))
    tx["split"] = "train"
    return tx


def test_features_on_toy_graph():
    f = build_features(toy_transactions()).set_index("tx_id")
    assert set(FEATURES) <= set(f.columns)
    assert f.loc[3, "log_amount_usd"] == np.log1p(55.0)          # EUR converted with the observed rate 1.1
    assert f.loc[1, "src_in_usd_24h"] == 100.0                     # B received 100 from A before paying C
    assert abs(f.loc[1, "passthrough_ratio"] - 0.9) < 1e-6
    assert f.loc[4, "reverse_pair_cnt_7d"] == 2                    # A->B happened twice before B->A
    assert f.loc[0, "reverse_pair_cnt_7d"] == 0
    assert f.loc[3, "src_out_cnt_7d"] == 2                         # A paid on day 1 and day 2
    assert f.loc[5, "src_distinct_dst_7d"] == 2                    # A paid B on day 1 and B on day 2 (1 + 1)
    assert f.loc[0, "src_distinct_dst_7d"] == 0                    # nothing visible before the first day ends


def test_no_future_leakage():
    tx = toy_transactions()
    full = build_features(tx).set_index("tx_id")
    early = build_features(tx[tx["ts"] < "2022-09-02"]).set_index("tx_id")
    cols = ["src_out_cnt_24h", "dst_in_usd_24h", "src_in_usd_24h", "reverse_pair_cnt_7d"]
    pd.testing.assert_frame_equal(full.loc[early.index, cols], early[cols], check_dtype=False)


def test_temporal_split_keeps_days_whole():
    ts = pd.Series(pd.to_datetime(["2022-09-01 10:00"] * 60 + ["2022-09-02 10:00"] * 20 + ["2022-09-03 10:00"] * 20))
    split = temporal_split(ts)
    assert (split[:60] == "train").all() and (split[60:80] == "val").all() and (split[80:] == "test").all()


def test_read_patterns_and_attach():
    import tempfile
    from pathlib import Path

    text = (
        "BEGIN LAUNDERING ATTEMPT - FAN-OUT:  Max 16-degree Fan-Out\n"
        "2022/09/01 01:00,B,C_acct,C,C_acct,90.00,US Dollar,90.00,US Dollar,Wire,1\n"
        "END LAUNDERING ATTEMPT - FAN-OUT\n"
    )
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "p.txt"
        p.write_text(text, encoding="utf-8")
        pat = read_patterns(p)
    assert pat.loc[0, "pattern"] == "FAN-OUT"
    tx = toy_transactions()
    tx.loc[1, ["src", "dst"]] = ["B_C_acct", "C_C_acct"]
    out = attach_patterns(tx, pat)
    assert out.loc[1, "pattern"] == "FAN-OUT"
    assert out.loc[2, "pattern"] == "UNCLASSIFIED"   # laundering but not in the patterns file
    assert out.loc[0, "pattern"] == "NONE"
