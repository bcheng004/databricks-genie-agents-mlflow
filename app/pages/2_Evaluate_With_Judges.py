"""Page 2 — Evaluate With LLM Judges (notebook 02).

Loads traces from the configured experiment into a selectable table, then runs
mlflow.genai.evaluate() on the chosen subset with a selection of built-in
judges, Guidelines judges, and code-based scorers.
"""

import json

import mlflow
import pandas as pd
import streamlit as st
from mlflow.entities import Feedback
from mlflow.genai.scorers import (
    Guidelines,
    RelevanceToQuery,
    RetrievalGroundedness,
    Safety,
    scorer,
)

from common import list_judge_models, require_experiment

st.title("⚖️ Evaluate Genie Agents")
st.caption(
    "Load the traces in your experiment, pick which ones to evaluate, then run "
    "`mlflow.genai.evaluate()` with built-in judges, Guidelines judges, and "
    "code-based scorers."
)

experiment_name, experiment_id = require_experiment()
st.info(f"Experiment: `{experiment_name}`")
mlflow.set_experiment(experiment_name)


# ---------------------------------------------------------------------------
# Helpers for the trace table
# ---------------------------------------------------------------------------
def _parse(val) -> dict:
    """Best-effort parse of a request/response cell into a dict."""
    if isinstance(val, str):
        try:
            return json.loads(val)
        except Exception:
            return {}
    return val if isinstance(val, dict) else {}


def _short(text, n: int = 120) -> str:
    if text is None:
        return ""
    s = str(text)
    return s if len(s) <= n else s[:n] + "…"


def build_display(df: pd.DataFrame) -> pd.DataFrame:
    """Build a human-friendly, one-row-per-trace table for selection.

    Uses MLflow 3.x column names (inputs/outputs/state/trace_id) and falls back
    to the 2.x names (request/response/status/request_id) if present.
    """
    cols = df.columns
    in_col = "inputs" if "inputs" in cols else "request"
    out_col = "outputs" if "outputs" in cols else "response"
    state_col = "state" if "state" in cols else "status"
    id_col = "trace_id" if "trace_id" in cols else "request_id"

    rows = []
    for _, r in df.iterrows():
        req = _parse(r.get(in_col))
        resp = _parse(r.get(out_col))
        rows.append(
            {
                "Question": _short(req.get("question"), 120),
                "Response": _short(resp.get("response"), 120),
                "SQL": "✓" if resp.get("generated_sql") else "",
                "Error": "✓" if resp.get("error") else "",
                "State": str(r.get(state_col, "")),
                "Trace ID": str(r.get(id_col, "")),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 1 — Load traces
# ---------------------------------------------------------------------------
st.subheader("Step 1 — Load traces")
st.caption("Loads all traces in the experiment.")

if st.button("Load traces"):
    with st.spinner("Loading traces…"):
        st.session_state["eval_traces_df"] = mlflow.search_traces(
            locations=[experiment_id],
            order_by=["timestamp DESC"],
            max_results=None,
        )

traces_df = st.session_state.get("eval_traces_df")
if traces_df is None:
    st.info("Click **Load traces** to fetch traces from your experiment.")
    st.stop()

if len(traces_df) == 0:
    st.warning(
        "No traces found in this experiment. "
        "Run **Trace Genie Conversations** first."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Step 2 — Select which traces to evaluate
# ---------------------------------------------------------------------------
st.subheader("Step 2 — Select traces to evaluate")
st.caption(
    f"Loaded **{len(traces_df)}** trace(s). Tick rows to evaluate a subset, or "
    "use the checkbox below to evaluate all of them."
)

display_df = build_display(traces_df)
event = st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True,
    on_select="rerun",
    selection_mode="multi-row",
    key="trace_table",
)

selected_rows = event.selection.rows if event and event.selection else []
eval_all = st.checkbox(f"Evaluate all {len(traces_df)} loaded traces (ignore selection)")

subset = traces_df if eval_all else traces_df.iloc[selected_rows]
n_selected = len(subset)
if eval_all:
    st.write(f"Evaluating **all {n_selected}** loaded trace(s).")
else:
    st.write(f"**{n_selected}** trace(s) selected.")

# ---------------------------------------------------------------------------
# Step 3 — Judges & scorers
# ---------------------------------------------------------------------------
st.subheader("Step 3 — Judges & scorers")
judge_model_options = ["Default"] + list_judge_models()


def _model_kwargs(choice: str) -> dict:
    return {} if choice == "Default" else {"model": f"databricks:/{choice}"}


st.markdown("**Built-in judges**")
builtin_judge_model = st.selectbox(
    "Model for built-in judges (Default uses each judge's built-in model)",
    judge_model_options,
    key="builtin_judge_model",
)
use_relevance = st.checkbox("RelevanceToQuery", value=True)
use_safety = st.checkbox("Safety", value=True)
use_groundedness = st.checkbox("RetrievalGroundedness", value=False)

st.markdown("**Guidelines judges (from notebook 02)**")
guideline_judge_model = st.selectbox(
    "Model for Guidelines judges (Default uses each judge's built-in model)",
    judge_model_options,
    key="guideline_judge_model",
)
use_response_quality = st.checkbox("genie_response_quality", value=True)
use_sql_quality = st.checkbox("genie_sql_quality", value=True)

st.subheader("Code-based scorers")
use_has_response = st.checkbox("has_response", value=True)
use_no_error = st.checkbox("no_error", value=True)

run = st.button("Run evaluation", type="primary", disabled=n_selected == 0)
if n_selected == 0:
    st.caption("Select at least one trace (or check *evaluate all*) to enable evaluation.")

if not run:
    st.stop()

# ---------------------------------------------------------------------------
# Build scorers
# ---------------------------------------------------------------------------
scorers = []
builtin_mkw = _model_kwargs(builtin_judge_model)
guideline_mkw = _model_kwargs(guideline_judge_model)

if use_relevance:
    scorers.append(RelevanceToQuery(**builtin_mkw))
if use_safety:
    scorers.append(Safety(**builtin_mkw))
if use_groundedness:
    scorers.append(RetrievalGroundedness(**builtin_mkw))

if use_response_quality:
    scorers.append(
        Guidelines(
            name="genie_response_quality",
            guidelines=[
                "The response must directly address the user's data question "
                "rather than giving a vague or generic reply.",
                "If SQL was generated, the response must include a data-driven "
                "answer, not just echo the SQL query back.",
                "The response must not say 'I cannot answer' when the question "
                "is about data that should be available in the tables.",
            ],
            **guideline_mkw,
        )
    )

if use_sql_quality:
    scorers.append(
        Guidelines(
            name="genie_sql_quality",
            guidelines=[
                "If SQL is present, it must use appropriate aggregation "
                "functions (SUM, COUNT, AVG) matching the user's intent.",
                "The SQL must include appropriate WHERE clauses to filter "
                "data as the user requested.",
                "The SQL must not use SELECT * on large tables without a "
                "LIMIT or specific filter.",
            ],
            **guideline_mkw,
        )
    )


@scorer
def has_response(outputs) -> Feedback:
    resp = outputs.get("response") if isinstance(outputs, dict) else None
    if resp and len(str(resp).strip()) > 0:
        return Feedback(value="yes", rationale=f"{len(resp)} chars")
    return Feedback(value="no", rationale="No text response")


@scorer
def no_error(outputs) -> Feedback:
    err = outputs.get("error") if isinstance(outputs, dict) else None
    if err and str(err).strip():
        return Feedback(value="no", rationale=f"Error: {str(err)[:200]}")
    return Feedback(value="yes", rationale="No errors")


if use_has_response:
    scorers.append(has_response)
if use_no_error:
    scorers.append(no_error)

if not scorers:
    st.error("Select at least one scorer before running evaluation.")
    st.stop()

st.write(f"Evaluating **{n_selected}** trace(s) with **{len(scorers)}** scorer(s)…")

with st.spinner("Running evaluation (this may take a few minutes)…"):
    try:
        result = mlflow.genai.evaluate(data=subset, scorers=scorers)
    except Exception as exc:
        st.error(f"Evaluation failed: {exc}")
        st.stop()

st.success("Evaluation complete!")


def _assessment_fields(a) -> tuple[str, object, str]:
    """Return (name, value, rationale) from an assessment, dict- or object-shaped.

    MLflow returns assessments as dicts (``a['assessment_name']``,
    ``a['feedback']['value']``) from ``search_traces`` DataFrames, but as objects
    (``a.name``, ``a.value``) in list form. Handle both defensively.
    """
    if isinstance(a, dict):
        name = a.get("assessment_name") or a.get("name") or "?"
        fb = a.get("feedback") or {}
        value = fb.get("value") if isinstance(fb, dict) else a.get("value")
        rationale = a.get("rationale") or (
            fb.get("rationale") if isinstance(fb, dict) else None
        )
    else:
        name = getattr(a, "name", None) or getattr(a, "assessment_name", None) or "?"
        value = getattr(a, "value", None)
        fb = getattr(a, "feedback", None)
        if value is None and fb is not None:
            value = getattr(fb, "value", None)
        rationale = getattr(a, "rationale", None)
    return str(name), value, ("" if rationale is None else str(rationale))


def build_score_tables(run_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (scores_df, rationales_df) with one row per evaluated trace.

    scores_df:     Question + one column per scorer (its value) + Trace ID.
    rationales_df: long form (Trace ID, Scorer, Value, Rationale) for detail.
    """
    df = mlflow.search_traces(run_id=run_id)

    cols = df.columns
    in_col = "inputs" if "inputs" in cols else "request"
    id_col = "trace_id" if "trace_id" in cols else "request_id"

    score_rows, rationale_rows = [], []
    for _, r in df.iterrows():
        req = _parse(r.get(in_col))
        trace_id = str(r.get(id_col, ""))
        question = _short(req.get("question"), 100)

        row = {"Question": question}
        for a in r.get("assessments") or []:
            name, value, rationale = _assessment_fields(a)
            row[name] = value
            rationale_rows.append(
                {
                    "Trace ID": trace_id,
                    "Scorer": name,
                    "Value": value,
                    "Rationale": rationale,
                }
            )
        row["Trace ID"] = trace_id
        score_rows.append(row)

    return pd.DataFrame(score_rows), pd.DataFrame(rationale_rows)


st.subheader("Per-trace scores")
try:
    scores_df, rationales_df = build_score_tables(result.run_id)
except Exception as exc:
    st.warning(f"Could not load per-trace results: {exc}")
    scores_df, rationales_df = pd.DataFrame(), pd.DataFrame()

if len(scores_df):
    st.caption("One row per trace; one column per scorer.")
    st.dataframe(scores_df, use_container_width=True, hide_index=True)
    with st.expander("Rationales (per trace × scorer)"):
        st.dataframe(rationales_df, use_container_width=True, hide_index=True)
else:
    st.info("No per-trace assessments were returned for this run.")

st.caption(
    f"Assessments written to experiment `{experiment_name}` — "
    "open the MLflow UI to explore trace-level results."
)
