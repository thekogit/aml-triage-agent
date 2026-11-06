"""HTTP API for the reviewer UI. Run: uvicorn aml.api:app --port 8000"""
from __future__ import annotations

from functools import lru_cache
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import config, db
from .agent import triage
from .investigate import alert_context
from .pii import Redactor
from .schema import AlertStatus
from .schema import Decision as DecisionSchema

app = FastAPI(
    title="aml-triage-agent",
    version="1.0.0",
    description="Transaction-monitoring alerts with an LLM investigation assistant. Synthetic data only.",
)


@lru_cache(maxsize=1)
def repo() -> db.PgRepo:
    return db.PgRepo(db.connect(config.PG_AGENT_DSN))


@lru_cache(maxsize=1)
def store() -> db.PgStore:
    return db.PgStore(db.connect(config.PG_DSN))


@lru_cache(maxsize=1)
def redactor() -> Redactor:
    names = [r["name"] for r in store().conn.execute("SELECT name FROM accounts").fetchall()]
    return Redactor(names)


class DecisionRequest(BaseModel):
    analyst: str = Field(..., min_length=2)
    action: Literal["APPROVE", "OVERRIDE_ESCALATE", "OVERRIDE_CLOSE"]
    comment: str = ""
    assessment_id: Optional[int] = None


@app.get("/health")
def health() -> dict:
    store().conn.execute("SELECT 1")
    return {"status": "ok", "model": config.LLM_MODEL}


@app.get("/alerts")
def list_alerts(status: AlertStatus = "OPEN", limit: int = 50) -> list[dict]:
    return store().list_alerts(status, min(limit, 500))


@app.get("/alerts/{alert_id}")
def get_alert(alert_id: int) -> dict:
    ctx = alert_context(repo(), alert_id)
    if "error" in ctx:
        raise HTTPException(404, ctx["error"])
    return {"alert": ctx, "latest_assessment": store().latest_assessment(alert_id)}


@app.post("/alerts/{alert_id}/triage")
def run_triage(alert_id: int) -> dict:
    try:
        run = triage(repo(), redactor(), alert_id)
    except ValueError as e:
        raise HTTPException(422, str(e))
    assessment_id = store().save_assessment(run)
    run_dict = dict(run)
    run_dict["assessment_id"] = assessment_id
    return run_dict


@app.post("/alerts/{alert_id}/decision")
def decide(alert_id: int, d: DecisionRequest) -> dict:
    if repo().alert(alert_id) is None:
        raise HTTPException(404, "alert not found")
    decision_model = DecisionSchema(
        analyst=d.analyst,
        action=d.action,
        comment=d.comment if d.comment else None,
    )
    decision_id = store().save_decision(alert_id, decision_model)
    return {"decision_id": decision_id, "alert_id": alert_id, "action": d.action}
