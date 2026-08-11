"""Page 3 — Improve Genie Agents (notebook 03).

Loads failed traces from the experiment, reads the Genie agent configuration,
and uses an LLM to generate specific, copy-paste-ready improvement suggestions.
"""

import json
import os
import re

import mlflow
import streamlit as st

from common import (
    genie_agent_id_input,
    get_openai_client,
    get_workspace_client,
    list_judge_models,
    require_experiment,
)


# Streamlit's st.markdown treats `$...$` as LaTeX math (KaTeX). Analyzer output is
# full of dollar signs (currency values, thresholds like "revenue over $1000"), so
# unescaped `$` pairs get swallowed and rendered as math. Escape `$` — but only
# outside code spans/blocks, where the markdown renderer already ignores it and a
# literal `\$` would otherwise show through in the code.
_CODE_SPAN_RE = re.compile(r"(```.*?```|~~~.*?~~~|`[^`\n]+`)", re.DOTALL)


def _escape_dollars_outside_code(text: str) -> str:
    """Escape `$` so it isn't rendered as LaTeX math, leaving code segments as-is."""
    parts = _CODE_SPAN_RE.split(text)
    # re.split with a capturing group keeps the matched code segments at odd
    # indices; escape only the non-code segments at even indices.
    return "".join(
        part.replace("$", "\\$") if i % 2 == 0 else part
        for i, part in enumerate(parts)
    )


st.title("🛠️ Improve Genie Agents")
st.caption(
    "Loads traces that failed evaluation, reads your Genie agent config, "
    "and asks an LLM to generate targeted improvement suggestions."
)

experiment_name, experiment_id = require_experiment()
st.info(f"Experiment: `{experiment_name}`")

space_id = genie_agent_id_input()

analyzer_model_options = list_judge_models()
_default_analyzer = os.environ.get("ANALYZER_MODEL", "databricks-claude-sonnet-5")
analyzer_model = st.selectbox(
    "Analyzer model",
    analyzer_model_options,
    index=(
        analyzer_model_options.index(_default_analyzer)
        if _default_analyzer in analyzer_model_options
        else 0
    ),
    help="Databricks Foundation Model API endpoint used to generate suggestions.",
)

max_failed = st.slider(
    "Max failed conversations to analyze", min_value=5, max_value=50, value=20, step=5
)

if not st.button("Generate improvement suggestions", type="primary"):
    st.stop()

if not space_id.strip():
    st.error("Please enter a Genie Agent ID.")
    st.stop()

w = get_workspace_client()
client = get_openai_client(w)
mlflow.set_experiment(experiment_name)

# Step 1: load failed traces
# Only Genie conversation traces are analyzable. The experiment also holds LLM
# judge calls (auto-traced by evaluation runs) and this page's own
# `analyze_genie_space` traces, none of which have a question to fix.
GENIE_TRACE_FILTER = "tags.`mlflow.traceName` = 'genie_interaction'"

with st.spinner("Loading traces with failures…"):
    all_traces = mlflow.search_traces(
        locations=[experiment_id],
        filter_string=GENIE_TRACE_FILTER,
        return_type="list",
    )

def _is_failure(assessment) -> bool:
    """True when an assessment represents a failed check.

    Scorers report failure in two shapes: the Guidelines/built-in judges use the
    string ``"no"``, while code-based scorers such as ``code_grounded`` return a
    plain ``False``. Checking only for ``"no"`` would silently drop the latter.
    """
    value = assessment.value
    if isinstance(value, bool):
        return value is False
    return str(value).strip().lower() == "no"


failed_conversations = []
for trace in all_traces:
    assessments = trace.info.assessments or []
    failures = [a for a in assessments if _is_failure(a)]
    if not failures:
        continue

    root = trace.data.spans[0]
    try:
        inputs = json.loads(root.inputs) if isinstance(root.inputs, str) else root.inputs
        outputs = json.loads(root.outputs) if isinstance(root.outputs, str) else root.outputs
    except Exception:
        inputs, outputs = {}, {}

    failed_conversations.append(
        {
            "question": inputs.get("question"),
            "response": outputs.get("response"),
            "generated_sql": outputs.get("generated_sql"),
            "error": outputs.get("error"),
            "failed_checks": [
                f"{a.name}: {a.value} — {a.rationale}" for a in failures
            ],
        }
    )

st.write(
    f"Found **{len(failed_conversations)}** failed conversation(s) out of "
    f"**{len(all_traces)}** total trace(s)."
)

if not failed_conversations:
    st.success("No failures found — your Genie agent is performing well on all evaluated traces!")
    st.stop()

# Step 2: read Genie agent config
with st.spinner("Reading Genie agent configuration…"):
    try:
        space = w.genie.get_space(space_id=space_id.strip(), include_serialized_space=True)
        config = json.loads(space.serialized_space) if space.serialized_space else {}
    except Exception as exc:
        st.error(f"Could not read Genie agent config: {exc}")
        st.stop()

tables = config.get("data_sources", {}).get("tables", [])
instructions = config.get("instructions", {})
text_instructions = instructions.get("text_instructions", [])
example_sqls = instructions.get("example_question_sqls", [])
table_names = [t["identifier"] for t in tables]

st.write(
    f"Agent: **{space.title}** — "
    f"{len(tables)} table(s), {len(text_instructions)} instruction(s), "
    f"{len(example_sqls)} example SQL(s)."
)

# Step 3: build prompts and call LLM
system_prompt = (
    "You are an expert Databricks AI/BI Genie agent consultant. "
    "You will be given conversations where Genie gave wrong or incomplete answers, "
    "along with the specific checks that failed. "
    "Generate specific, copy-paste-ready fixes: SQL expressions, text instructions, "
    "example SQL, and column descriptions. "
    "Never give vague advice. Always write the actual implementation. "
    "Format every SQL statement and identifier inside a fenced ```sql code block "
    "so it renders correctly."
)

analysis_prompt = f"""Fix the issues found in these Genie conversations.

## FAILED CONVERSATIONS
{json.dumps(failed_conversations[:max_failed], indent=2)}

## CURRENT AGENT CONFIG
Title: {space.title}
Tables: {', '.join(table_names[:10])}
Text instructions: {len(text_instructions)}
Example SQL: {len(example_sqls)}

For each failed conversation, provide a specific fix: a new text instruction,
SQL expression, example query, or column description that would prevent the failure.
Prioritize by impact."""


def _content_to_text(content) -> str:
    """Coerce a chat completion's ``message.content`` to plain text.

    Most endpoints return a string, but some (e.g. Claude models that emit
    structured content) return a list of content-part dicts/objects such as
    ``[{"type": "text", "text": "..."}]``. Concatenate the text parts so the
    rest of the page (dollar-escaping, ``st.markdown``) always sees a ``str``.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(item.get("text") or item.get("content") or "")
            else:
                parts.append(getattr(item, "text", "") or "")
        return "".join(parts)
    return str(content)


@mlflow.trace
def analyze_genie_agent(user_prompt: str, sys_prompt: str) -> str:
    response = client.chat.completions.create(
        model=analyzer_model,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=8000,
    )
    return _content_to_text(response.choices[0].message.content)


with st.spinner("Asking the LLM to generate improvement suggestions…"):
    try:
        recommendations = analyze_genie_agent(analysis_prompt, system_prompt)
    except Exception as exc:
        st.error(f"LLM call failed: {exc}")
        st.stop()

st.subheader("Improvement Suggestions")
st.markdown(_escape_dollars_outside_code(recommendations))
st.caption("This analysis was traced to your MLflow experiment for future reference.")
