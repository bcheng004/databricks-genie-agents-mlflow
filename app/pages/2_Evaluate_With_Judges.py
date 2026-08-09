"""Page 2 — Evaluate With LLM Judges (notebook 02).

Loads traces from the configured experiment into a selectable table, then runs
mlflow.genai.evaluate() on the chosen subset with a selection of built-in
judges, Guidelines judges, and code-based scorers.
"""

import json
import re

import mlflow
import pandas as pd
import streamlit as st
from mlflow.entities import AssessmentError, Feedback
from mlflow.genai.scorers import (
    Guidelines,
    RelevanceToQuery,
    Safety,
)

from code_search import (
    MAX_RATIONALE_SNIPPET_CHARS,
    NO_RESULTS,
    CodeSearcher,
    format_rows,
    source_links,
)
from common import (
    get_guideline_judges,
    get_openai_client,
    get_workspace_client,
    list_judge_models,
    require_experiment,
)

# Preferred default endpoint for the code_grounded judge (falls back to the
# first available model when this endpoint isn't listed).
DEFAULT_CODE_GROUNDED_MODEL = "databricks-gpt-5-6-luna"

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
# Genie conversation traces are logged by page 1 under this span name. The
# experiment also collects traces that are *not* Genie conversations and make no
# sense to evaluate: mlflow.genai.evaluate() enables autologging, so every LLM
# judge call lands here as a `DatabricksCompletions` trace, and the Improve page
# logs `analyze_genie_space`. Those have no `question` input, so they'd show up
# as blank rows and score meaninglessly — filter them out at search time.
GENIE_TRACE_NAME = "genie_interaction"
GENIE_TRACE_FILTER = f"tags.`mlflow.traceName` = '{GENIE_TRACE_NAME}'"

st.subheader("Step 1 — Load traces")
st.caption("Loads Genie conversation traces from your experiment.")

if st.button("Load traces"):
    with st.spinner("Loading traces…"):
        st.session_state["eval_traces_df"] = mlflow.search_traces(
            locations=[experiment_id],
            filter_string=GENIE_TRACE_FILTER,
            order_by=["timestamp DESC"],
            max_results=None,
        )

traces_df = st.session_state.get("eval_traces_df")
if traces_df is None:
    st.info("Click **Load traces** to fetch traces from your experiment.")
    st.stop()

if len(traces_df) == 0:
    st.warning(
        "No Genie conversation traces found in this experiment. "
        "Run **Trace Conversations** first."
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

st.markdown("**Guidelines judges**")
guideline_judge_model = st.selectbox(
    "Model for Guidelines judges (Default uses each judge's built-in model)",
    judge_model_options,
    key="guideline_judge_model",
)
guideline_judges = get_guideline_judges()
if guideline_judges:
    selected_guidelines = {
        name: st.checkbox(name, value=True, key=f"guideline_{name}")
        for name in guideline_judges
    }
else:
    selected_guidelines = {}
    st.caption(
        "No guideline judges yet — add some on the **Create Guideline Judge** page."
    )

st.subheader("Code-based scorers")
use_code_grounded = st.checkbox(
    "code_grounded",
    value=False,
    help=(
        "Retrieves code from the Lakebase BM25 code index for each question, then "
        "asks an LLM whether the response is grounded in that code."
    ),
)
if use_code_grounded:
    code_grounded_options = list_judge_models()
    code_grounded_default_index = (
        code_grounded_options.index(DEFAULT_CODE_GROUNDED_MODEL)
        if DEFAULT_CODE_GROUNDED_MODEL in code_grounded_options
        else 0
    )
    code_grounded_model = st.selectbox(
        "Model for code_grounded judge",
        code_grounded_options,
        index=code_grounded_default_index,
        key="code_grounded_model",
        help="Foundation Model endpoint used to judge groundedness against the retrieved code.",
    )
    code_grounded_top_k = st.slider(
        "Code snippets to retrieve per question",
        min_value=1,
        max_value=10,
        value=5,
        key="code_grounded_top_k",
    )
    with st.expander("Lakebase connection"):
        try:
            st.json(CodeSearcher(get_workspace_client()).connection_info())
        except Exception as exc:
            st.warning(f"Could not resolve the Lakebase endpoint: {exc}")
else:
    code_grounded_model = None
    code_grounded_top_k = 5

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

for name, selected in selected_guidelines.items():
    if selected:
        scorers.append(
            Guidelines(
                name=name,
                guidelines=guideline_judges[name],
                **guideline_mkw,
            )
        )


# --- code_grounded ---------------------------------------------------------
# The judge wording is deliberately explicit: a looser "is this based on the
# snippets?" prompt votes yes on answers whose own rationale calls the API
# fabricated. The labelled VERDICT line is what _verdict_of parses.
_CODE_JUDGE_PROMPT = (
    "You are an evaluation judge assessing whether an answer is grounded in "
    "reference code snippets.\n\n"
    "Answer 'yes' only if the substantive technical claims in the answer (APIs, "
    "imports, function and class names, usage patterns) are supported by the "
    "snippets.\n"
    "Answer 'no' if the answer invents APIs or patterns that do not appear in "
    "the snippets, or contradicts them.\n\n"
    "Respond in exactly this format:\n"
    "VERDICT: yes|no\n"
    "RATIONALE: one or two sentences naming the specific APIs or patterns that "
    "support your verdict."
)

# Reasoning models spend the budget on hidden reasoning before any answer text;
# below ~1000 they return reasoning-only with finish_reason="length".
_CODE_JUDGE_MAX_TOKENS = 2000

_VERDICT_RE = re.compile(r"^\s*(?:verdict\s*:\s*)?(yes|no)\b", re.IGNORECASE)


def _judge_text_of(content) -> str:
    """Flatten a completion's content, keeping only answer text.

    Reasoning models return a list of blocks; the verdict lives in the ``text``
    blocks, while ``reasoning`` blocks carry an opaque (usually empty) summary.
    """
    if isinstance(content, list):
        return "".join(
            block.get("text", "") or ""
            for block in content
            if isinstance(block, dict) and block.get("type") != "reasoning"
        )
    return "" if content is None else str(content)


def _verdict_of(judge_text: str) -> tuple[bool | None, str]:
    """Return ``(verdict, rationale)``, or ``(None, ...)`` if there is no verdict."""
    lines = judge_text.splitlines()
    for i, line in enumerate(lines):
        match = _VERDICT_RE.match(line)
        if not match:
            continue
        verdict = match.group(1).lower() == "yes"
        rest = _VERDICT_RE.sub("", line, count=1).strip(" :.-")
        tail = "\n".join(lines[i + 1 :]).strip()
        rationale = "\n".join(p for p in (rest, tail) if p).strip()
        rationale = re.sub(r"^rationale\s*:\s*", "", rationale, flags=re.IGNORECASE)
        return verdict, (rationale or judge_text)
    return None, judge_text


def _run_code_grounded(inputs, outputs) -> Feedback:
    """Score one trace against the Lakebase BM25 code index.

    Returns a ``Feedback``; the caller writes it with ``mlflow.log_feedback``
    so the assessment lands directly on the existing trace rather than via an
    evaluation run. Module-level ``code_searcher``, ``code_judge_client``,
    ``code_judge_model``, and ``code_top_k`` are set before this is called.
    """
    question = inputs.get("question") or inputs.get("query") or str(inputs)

    try:
        rows = code_searcher.search_rows(question, top_k=code_top_k)
    except Exception as exc:
        # An unreachable index is a scorer failure, not evidence the answer was
        # ungrounded — record it as an error, not a "no".
        return Feedback(
            error=AssessmentError(
                error_code="CODE_SEARCH_UNAVAILABLE",
                error_message=f"{type(exc).__name__}: {exc}",
            )
        )

    if not rows:
        return Feedback(value="no", rationale=NO_RESULTS)

    snippets = format_rows(rows)

    try:
        response = code_judge_client.chat.completions.create(
            model=code_judge_model,
            messages=[
                {"role": "system", "content": _CODE_JUDGE_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Question: {question}\n\n"
                        f"Answer: {outputs}\n\n"
                        f"Reference code snippets:\n{snippets}"
                    ),
                },
            ],
            max_tokens=_CODE_JUDGE_MAX_TOKENS,
        )
    except Exception as exc:
        return Feedback(
            error=AssessmentError(
                error_code="JUDGE_CALL_FAILED",
                error_message=f"{type(exc).__name__}: {exc}",
            )
        )

    choice = response.choices[0]
    judge_text = _judge_text_of(choice.message.content).strip()
    if not judge_text:
        return Feedback(
            error=AssessmentError(
                error_code="JUDGE_NO_VERDICT",
                error_message=(
                    f"Judge returned no answer text "
                    f"(finish_reason={choice.finish_reason!r}). Try a higher "
                    f"max_tokens or a non-reasoning judge model."
                ),
            )
        )

    verdict, rationale = _verdict_of(judge_text)
    if verdict is None:
        return Feedback(
            error=AssessmentError(
                error_code="JUDGE_UNPARSEABLE_VERDICT",
                error_message=f"No yes/no verdict in judge output: {judge_text[:300]!r}",
            )
        )

    shown = snippets[:MAX_RATIONALE_SNIPPET_CHARS]
    if len(snippets) > MAX_RATIONALE_SNIPPET_CHARS:
        shown += "\n… (truncated)"

    # "yes"/"no" rather than a bool, matching the judges above: MLflow groups
    # assessments by value type, so a bool renders apart from the other scorers.
    return Feedback(
        value="yes" if verdict else "no",
        rationale=(
            f"{rationale}\n\n"
            f"--- Sources ---\n{chr(10).join(source_links(rows))}\n\n"
            f"--- Retrieved Code Snippets ---\n{shown}"
        ),
    )


# code_grounded runs separately via log_feedback (not inside evaluate()) so it
# never spawns an evaluation run and never auto-traces judge calls back into the
# experiment. The connection objects are built here on the main thread where the
# OBO token is readable; the scorer reads them as module-level globals at call time.
if use_code_grounded:
    try:
        _w = get_workspace_client()
        code_searcher = CodeSearcher(_w)
        code_judge_client = get_openai_client(_w)
        code_judge_model = code_grounded_model
        code_top_k = code_grounded_top_k
    except Exception as exc:
        st.error(f"Could not set up the code_grounded scorer: {exc}")
        st.stop()

# Decide whether we have anything to run through evaluate() (everything except
# code_grounded) and/or the direct log_feedback loop (only code_grounded).
run_evaluate = bool(scorers)
run_code_grounded = use_code_grounded

if not run_evaluate and not run_code_grounded:
    st.error("Select at least one scorer before running evaluation.")
    st.stop()

n_scorers = len(scorers) + (1 if run_code_grounded else 0)
st.write(f"Evaluating **{n_selected}** trace(s) with **{n_scorers}** scorer(s)…")

eval_run_id = None
if run_evaluate:
    with st.spinner("Running evaluation (this may take a few minutes)…"):
        try:
            result = mlflow.genai.evaluate(data=subset, scorers=scorers)
            eval_run_id = result.run_id
        except Exception as exc:
            st.error(f"Evaluation failed: {exc}")
            st.stop()

if run_code_grounded:
    cols = subset.columns
    in_col = "inputs" if "inputs" in cols else "request"
    out_col = "outputs" if "outputs" in cols else "response"
    id_col = "trace_id" if "trace_id" in cols else "request_id"

    progress = st.progress(0.0, text="Scoring with code_grounded…")
    n = len(subset)
    n_ok, n_err = 0, 0
    for i, (_, row) in enumerate(subset.iterrows()):
        trace_id = str(row.get(id_col, ""))
        inputs = _parse(row.get(in_col))
        outputs = _parse(row.get(out_col))
        fb = _run_code_grounded(inputs, outputs)
        try:
            mlflow.log_feedback(
                trace_id=trace_id,
                name="code_grounded",
                value=fb.feedback.value if fb.error is None else None,
                error=fb.error,
                rationale=fb.rationale,
            )
            n_ok += 1
        except Exception as exc:
            st.warning(f"Could not write feedback for trace {trace_id}: {exc}")
            n_err += 1
        progress.progress((i + 1) / n, text=f"code_grounded: {i + 1}/{n}")
    progress.empty()
    if n_err:
        st.warning(f"code_grounded: {n_ok} written, {n_err} failed.")
    else:
        st.success(f"code_grounded written to **{n_ok}** trace(s).")

if run_evaluate:
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


def build_score_tables(eval_df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per (trace × scorer) with Question, Scorer, Value, Rationale.

    Only the latest assessment per scorer is kept (highest create_time_ms) so
    re-running evaluation doesn't show stale results alongside new ones.
    """
    cols = eval_df.columns
    in_col = "inputs" if "inputs" in cols else "request"
    id_col = "trace_id" if "trace_id" in cols else "request_id"

    rows = []
    for _, r in eval_df.iterrows():
        req = _parse(r.get(in_col))
        trace_id = str(r.get(id_col, ""))
        question = _short(req.get("question"), 100)

        # Keep only the latest assessment per scorer name.
        latest: dict[str, tuple] = {}  # scorer_name → (create_time_ms, assessment)
        for a in r.get("assessments") or []:
            name = (
                a.get("assessment_name") or a.get("name")
                if isinstance(a, dict)
                else getattr(a, "name", None) or getattr(a, "assessment_name", None)
            ) or "?"
            ts = (
                a.get("create_time_ms", 0)
                if isinstance(a, dict)
                else getattr(a, "create_time_ms", 0) or 0
            )
            if name not in latest or ts > latest[name][0]:
                latest[name] = (ts, a)

        for name, (_, a) in sorted(latest.items()):
            _, value, rationale = _assessment_fields(a)
            rows.append(
                {
                    "Question": question,
                    "Scorer": name,
                    "Value": value,
                    "Rationale": rationale,
                    "Trace ID": trace_id,
                }
            )

    return pd.DataFrame(rows)


st.subheader("Per-trace scores")
try:
    # Re-fetch the evaluated traces so log_feedback() assessments are included.
    id_col = "trace_id" if "trace_id" in subset.columns else "request_id"
    evaluated_ids = [str(v) for v in subset[id_col]]
    refreshed = mlflow.search_traces(
        locations=[experiment_id],
        filter_string=GENIE_TRACE_FILTER,
        order_by=["timestamp DESC"],
        max_results=None,
    )
    # Keep only the rows we just evaluated.
    refreshed = refreshed[refreshed[id_col].astype(str).isin(set(evaluated_ids))]
    results_df = build_score_tables(refreshed)
except Exception as exc:
    st.warning(f"Could not load per-trace results: {exc}")
    results_df = pd.DataFrame()

if len(results_df):
    st.caption("One row per trace × scorer, with rationale inline.")
    # Render as an HTML table so the Rationale column wraps with a hard cap.
    _RATIONALE_MAX = 400

    def _to_html(df: pd.DataFrame) -> str:
        import html
        cols = list(df.columns)
        header = "".join(
            f'<th style="padding:6px 10px;border-bottom:1px solid #ccc;'
            f'white-space:nowrap;font-weight:600">{html.escape(c)}</th>'
            for c in cols
        )
        rows_html = ""
        for _, row in df.iterrows():
            cells = ""
            for c in cols:
                raw = "" if row[c] is None else str(row[c])
                if c == "Rationale":
                    val = raw[:_RATIONALE_MAX] + ("…" if len(raw) > _RATIONALE_MAX else "")
                    style = (
                        "min-width:340px;max-width:480px;"
                        "word-break:break-word;white-space:pre-wrap"
                    )
                else:
                    val = raw
                    style = "white-space:nowrap"
                cells += (
                    f'<td style="padding:6px 10px;border-bottom:1px solid #eee;'
                    f'vertical-align:top;{style}">{html.escape(val)}</td>'
                )
            rows_html += f"<tr>{cells}</tr>"
        return (
            '<div style="overflow-x:auto">'
            '<table style="width:100%;border-collapse:collapse;font-size:0.85rem">'
            f"<thead><tr>{header}</tr></thead>"
            f"<tbody>{rows_html}</tbody>"
            "</table></div>"
        )

    st.html(_to_html(results_df))
else:
    st.info("No per-trace assessments were returned for this run.")

st.caption(
    f"Assessments written to experiment `{experiment_name}` — "
    "open the MLflow UI to explore trace-level results."
)
