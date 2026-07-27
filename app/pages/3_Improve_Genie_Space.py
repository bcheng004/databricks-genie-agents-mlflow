"""Page 3 — Improve Genie Space (notebook 03).

Loads failed traces from the experiment, reads the Genie Space configuration,
and uses an LLM to generate specific, copy-paste-ready improvement suggestions.
"""

import json
import os

import mlflow
import streamlit as st

from common import (
    genie_space_id_input,
    get_openai_client,
    get_workspace_client,
    list_judge_models,
    require_experiment,
)

st.title("🛠️ Improve Genie Space")
st.caption(
    "Loads traces that failed evaluation, reads your Genie Space config, "
    "and asks an LLM to generate targeted improvement suggestions."
)

experiment_name, experiment_id = require_experiment()
st.info(f"Experiment: `{experiment_name}`")

space_id = genie_space_id_input()

analyzer_model_options = list_judge_models()
_default_analyzer = os.environ.get("ANALYZER_MODEL", "databricks-claude-sonnet-4-6")
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
    st.error("Please enter a Genie Space ID.")
    st.stop()

w = get_workspace_client()
client = get_openai_client(w)
mlflow.set_experiment(experiment_name)

# Step 1: load failed traces
with st.spinner("Loading traces with failures…"):
    all_traces = mlflow.search_traces(
        locations=[experiment_id], return_type="list"
    )

failed_conversations = []
for trace in all_traces:
    assessments = trace.info.assessments or []
    failures = [a for a in assessments if a.value == "no"]
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
    st.success("No failures found — your Genie Space is performing well on all evaluated traces!")
    st.stop()

# Step 2: read Genie Space config
with st.spinner("Reading Genie Space configuration…"):
    try:
        space = w.genie.get_space(space_id=space_id.strip(), include_serialized_space=True)
        config = json.loads(space.serialized_space) if space.serialized_space else {}
    except Exception as exc:
        st.error(f"Could not read Genie Space config: {exc}")
        st.stop()

tables = config.get("data_sources", {}).get("tables", [])
instructions = config.get("instructions", {})
text_instructions = instructions.get("text_instructions", [])
example_sqls = instructions.get("example_question_sqls", [])
table_names = [t["identifier"] for t in tables]

st.write(
    f"Space: **{space.title}** — "
    f"{len(tables)} table(s), {len(text_instructions)} instruction(s), "
    f"{len(example_sqls)} example SQL(s)."
)

# Step 3: build prompts and call LLM
system_prompt = (
    "You are an expert Databricks AI/BI Genie space consultant. "
    "You will be given conversations where Genie gave wrong or incomplete answers, "
    "along with the specific checks that failed. "
    "Generate specific, copy-paste-ready fixes: SQL expressions, text instructions, "
    "example SQL, and column descriptions. "
    "Never give vague advice. Always write the actual implementation."
)

analysis_prompt = f"""Fix the issues found in these Genie conversations.

## FAILED CONVERSATIONS
{json.dumps(failed_conversations[:max_failed], indent=2)}

## CURRENT SPACE CONFIG
Title: {space.title}
Tables: {', '.join(table_names[:10])}
Text instructions: {len(text_instructions)}
Example SQL: {len(example_sqls)}

For each failed conversation, provide a specific fix: a new text instruction,
SQL expression, example query, or column description that would prevent the failure.
Prioritize by impact."""


@mlflow.trace
def analyze_genie_space(user_prompt: str, sys_prompt: str) -> str:
    response = client.chat.completions.create(
        model=analyzer_model,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=8000,
        temperature=0.1,
    )
    return response.choices[0].message.content


with st.spinner("Asking the LLM to generate improvement suggestions…"):
    try:
        recommendations = analyze_genie_space(analysis_prompt, system_prompt)
    except Exception as exc:
        st.error(f"LLM call failed: {exc}")
        st.stop()

st.subheader("Improvement Suggestions")
st.markdown(recommendations)
st.caption("This analysis was traced to your MLflow experiment for future reference.")
