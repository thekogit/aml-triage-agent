"""Reviewer UI. Run: streamlit run aml/ui.py  (needs the API running)."""
from __future__ import annotations

import os

import pandas as pd
import requests
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="AML alert review", layout="wide")
st.title("AML alert review")
st.caption("Synthetic data (IBM AMLworld). The assistant recommends; the analyst decides.")

status = st.radio("Alerts", ["OPEN", "TRIAGED", "DECIDED"], horizontal=True)
try:
    resp = requests.get(f"{API}/alerts", params={"status": status, "limit": 100}, timeout=30)
    alerts = resp.json() if resp.ok else []
except Exception as e:
    st.error(f"Cannot connect to API at {API}: {e}")
    st.stop()

if not alerts:
    st.info("No alerts with this status.")
    st.stop()

st.dataframe(pd.DataFrame(alerts), use_container_width=True, hide_index=True)

alert_id = st.selectbox("Alert", [a["alert_id"] for a in alerts])
try:
    detail = requests.get(f"{API}/alerts/{alert_id}", timeout=30).json()
except Exception as e:
    st.error(f"Failed to fetch alert details: {e}")
    st.stop()

left, right = st.columns(2)
with left:
    st.subheader("Alert")
    st.json(detail.get("alert", {}))

with right:
    st.subheader("Assistant assessment")
    if st.button("Run investigation", type="primary"):
        with st.spinner("Investigating (tools are called live)…"):
            try:
                res = requests.post(f"{API}/alerts/{alert_id}/triage", timeout=600)
                if res.ok:
                    st.session_state["run"] = res.json()
                else:
                    st.error(f"Triage failed: {res.text}")
            except Exception as e:
                st.error(f"Triage request failed: {e}")

    run = st.session_state.get("run")
    if run and run.get("alert_id") == alert_id:
        fa = run["final_assessment"]
        st.metric("Recommendation", fa["recommendation"], help="after deterministic guard")
        if run.get("guard_reasons"):
            st.warning("Guard changed the LLM's recommendation: " + "; ".join(run["guard_reasons"]))
        st.write(fa["narrative"])
        grounded_pct = run.get("grounding", {}).get("grounded_rate", 0.0)
        st.write(
            f"Typology: **{fa['typology']}** · risk: **{fa['risk_level']}** · "
            f"grounded evidence: **{grounded_pct:.0%}**"
        )
        st.dataframe(pd.DataFrame(fa.get("evidence", [])), hide_index=True)
        with st.expander("Tool calls (audit trail)"):
            for e in run.get("tool_log", []):
                st.code(f"[{e['tool_call_id']}] {e['name']}({e['args']})\n{str(e['output'])[:3000]}")

st.subheader("Decision")
with st.form("decision"):
    analyst = st.text_input("Analyst id")
    action = st.selectbox("Action", ["APPROVE", "OVERRIDE_ESCALATE", "OVERRIDE_CLOSE"])
    comment = st.text_area("Comment (required for overrides)")
    if st.form_submit_button("Submit decision"):
        if action.startswith("OVERRIDE") and not comment.strip():
            st.error("Overrides need a comment.")
        elif not analyst.strip():
            st.error("Analyst id is required.")
        else:
            body = {
                "analyst": analyst,
                "action": action,
                "comment": comment,
                "assessment_id": (st.session_state.get("run") or {}).get("assessment_id"),
            }
            try:
                r = requests.post(f"{API}/alerts/{alert_id}/decision", json=body, timeout=30)
                if r.ok:
                    st.success(f"Saved decision {r.json().get('decision_id')}")
                else:
                    st.error(r.text)
            except Exception as e:
                st.error(f"Failed to submit decision: {e}")
