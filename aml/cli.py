"""Command line entry point. Run `python -m aml.cli --help`."""
from __future__ import annotations

import argparse
import json

import pandas as pd

from . import config


def cmd_ingest(a):
    from .ingest import attach_patterns, read_patterns, read_transactions, temporal_split

    tx = read_transactions(config.RAW_TRANS, nrows=a.nrows)
    if config.RAW_PATTERNS.exists():
        tx = attach_patterns(tx, read_patterns(config.RAW_PATTERNS))
    else:
        print("WARNING: patterns file not found; typology accuracy will be N/A")
    tx["split"] = temporal_split(tx["ts"])
    tx.to_parquet(config.TX_PATH, index=False)
    print(
        tx.groupby("split")
        .agg(n=("tx_id", "size"), laundering=("is_laundering", "sum"), first=("ts", "min"), last=("ts", "max"))
        .to_string()
    )


def cmd_features(a):
    from .features import build_features

    f = build_features(pd.read_parquet(config.TX_PATH), config.USD_NAME)
    f.to_parquet(config.FEAT_PATH, index=False)
    print(f"{len(f)} rows x {f.shape[1]} columns -> {config.FEAT_PATH}")
    print(f[["src_out_cnt_24h", "dst_in_cnt_24h", "passthrough_ratio", "reverse_pair_cnt_7d"]].describe().T.to_string())


def cmd_train(a):
    from .model import evaluate_and_alert, save_report, train

    feat = pd.read_parquet(config.FEAT_PATH)
    clf, categories = train(feat)
    config.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    clf.booster_.save_model(str(config.MODEL_PATH))
    (config.MODEL_PATH.parent / "categories.json").write_text(json.dumps(categories), encoding="utf-8")
    out = evaluate_and_alert(feat, clf, categories)
    save_report(out, config.RESULTS)
    out["alerts"].to_parquet(config.ALERTS_PATH, index=False)
    out["truth"].to_parquet(config.TRUTH_PATH, index=False)
    out["scores"].to_parquet(config.SCORES_PATH, index=False)
    print((config.RESULTS / "model_eval.md").read_text(encoding="utf-8"))


def cmd_load_db(a):
    from . import db
    from .loader import load_all

    tx = pd.read_parquet(config.TX_PATH)
    usd = pd.read_parquet(config.FEAT_PATH, columns=["tx_id", "amount_usd"]).set_index("tx_id")["amount_usd"]
    conn = db.connect()
    counts = load_all(conn, tx, usd, pd.read_parquet(config.ALERTS_PATH), pd.read_parquet(config.TRUTH_PATH))
    print(counts)


def cmd_index_policies(a):
    from . import db
    from .rag import index_policies

    print(f"{index_policies(db.connect())} policy chunks indexed")


def _repo_and_redactor():
    from . import db
    from .pii import Redactor

    owner_conn = db.connect(config.PG_DSN)
    names = [r["name"] for r in owner_conn.execute("SELECT name FROM accounts").fetchall()]
    agent_conn = db.connect(config.PG_AGENT_DSN)
    repo = db.PgRepo(agent_conn)
    return repo, Redactor(names)


def cmd_triage(a):
    from .agent import triage

    repo, red = _repo_and_redactor()
    run = triage(repo, red, a.alert_id)
    print(
        json.dumps(
            {
                k: run[k]
                for k in ["final_assessment", "guard_reasons", "grounding", "tool_calls", "latency_s", "cost_usd"]
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def cmd_eval_agent(a):
    from . import db
    from .agent import triage
    from .evaluate_agent import run_eval

    repo, red = _repo_and_redactor()
    store = db.PgStore(db.connect(config.PG_DSN))
    print(json.dumps(run_eval(repo, store, red, a.n, config.RESULTS, triage), indent=2))


def cmd_injection_test(a):
    from . import db
    from .agent import triage
    from .evaluate_agent import injection_test

    repo, red = _repo_and_redactor()
    store = db.PgStore(db.connect(config.PG_DSN))
    df = injection_test(repo, store, red, a.n, config.RESULTS, triage)
    print(df.to_string(index=False))
    print(f"flipped decisions: {int(df['flipped'].sum())} of {len(df)}")


def cmd_judge(a):
    from .agent import make_client
    from .judge import agreement, judge_all

    j = judge_all(make_client(), config.RESULTS / "agent_runs.jsonl", config.RESULTS / "judge_scores.csv")
    print(j[["faithfulness", "usefulness"]].describe().to_string())
    if config.HUMAN_LABELS.exists():
        print(json.dumps(agreement(j, pd.read_csv(config.HUMAN_LABELS)), indent=2))
    else:
        print(f"no {config.HUMAN_LABELS} yet: score 30 notes by hand to measure agreement")


EXPECTED_HEADER = (
    "Timestamp,From Bank,Account,To Bank,Account,Amount Received,Receiving Currency,"
    "Amount Paid,Payment Currency,Payment Format,Is Laundering"
)


def cmd_check_raw(a):
    header = config.RAW_TRANS.open(encoding="utf-8-sig").readline().strip()
    print(f"transactions header ok: {header == EXPECTED_HEADER}")
    if header != EXPECTED_HEADER:
        print(f"  found:    {header}\n  expected: {EXPECTED_HEADER}")
    if config.RAW_PATTERNS.exists():
        first = config.RAW_PATTERNS.open(encoding="utf-8-sig").readline().strip()
        print(f"patterns first line ok: {first.startswith('BEGIN LAUNDERING ATTEMPT')}  ({first[:60]})")
    else:
        print("patterns file missing")


def cmd_db_check(a):
    from . import db

    conn = db.connect()
    for t in ["transactions", "accounts", "alerts", "alert_truth", "policy_chunks", "assessments", "decisions"]:
        n = conn.execute(f"SELECT count(*) AS n FROM {t}").fetchone()["n"]
        print(f"{t:15s} {n}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="aml")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check-raw", help="check the Kaggle files' format").set_defaults(func=cmd_check_raw)
    sub.add_parser("db-check", help="row counts of every table").set_defaults(func=cmd_db_check)
    s = sub.add_parser("ingest", help="Kaggle CSV -> data/transactions.parquet")
    s.add_argument("--nrows", type=int, default=None, help="read only the first N rows (smoke test)")
    s.set_defaults(func=cmd_ingest)
    sub.add_parser("features", help="-> data/features.parquet").set_defaults(func=cmd_features)
    sub.add_parser("train", help="train LightGBM, write alerts and results/model_eval.md").set_defaults(func=cmd_train)
    sub.add_parser("load-db", help="load the serving subset into Postgres").set_defaults(func=cmd_load_db)
    sub.add_parser("index-policies", help="embed aml/policies into pgvector").set_defaults(func=cmd_index_policies)
    s = sub.add_parser("triage", help="investigate one alert and print the assessment")
    s.add_argument("--alert-id", type=int, required=True)
    s.set_defaults(func=cmd_triage)
    s = sub.add_parser("eval-agent", help="run the agent on a balanced sample of alerts")
    s.add_argument("--n", type=int, default=200)
    s.set_defaults(func=cmd_eval_agent)
    s = sub.add_parser("injection-test", help="hostile memo must not flip decisions")
    s.add_argument("--n", type=int, default=10)
    s.set_defaults(func=cmd_injection_test)
    sub.add_parser("judge", help="LLM-as-judge on results/agent_runs.jsonl").set_defaults(func=cmd_judge)
    a = p.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
