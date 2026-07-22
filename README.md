# Databricks Genie + MLflow Evaluation

Evaluate and improve [Databricks Genie](https://docs.databricks.com/genie/) spaces
(text-to-SQL assistants) using MLflow. Genie conversations become MLflow traces you
can inspect, score with LLM judges, and turn into concrete configuration fixes.

Based on the [MLflow Databricks Genie cookbook](https://mlflow.org/cookbook/databricks-genie/).

## Prerequisites

- A Databricks workspace with an existing Genie space that has conversations
- Python environment authenticated to Databricks (`databricks-sdk` picks up your config)

```bash
pip install "mlflow[genai]" databricks-sdk openai
```

## Pipeline

The workflow runs in three sequential stages.

### 1. Trace conversations

Extract Genie conversations and log each as an MLflow trace.

```python
import mlflow
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
SPACE_ID = "your-genie-space-id"
EXPERIMENT_NAME = f"/Users/{w.current_user.me().user_name}/genie_eval"
mlflow.set_experiment(EXPERIMENT_NAME)

for convo in w.genie.list_conversations(space_id=SPACE_ID, include_all=True):
    for msg in w.genie.list_conversation_messages(
        space_id=SPACE_ID, conversation_id=convo.conversation_id
    ):
        with mlflow.start_span(name="genie_interaction") as span:
            span.set_inputs({"question": msg.content})
            span.set_outputs({"sql": ..., "response": ...})
            mlflow.update_current_trace(tags={"message_id": msg.message_id})
```

Tagging with `message_id` keeps re-runs idempotent — check `mlflow.search_traces(...)`
first to skip messages already logged.

### 2. Evaluate with LLM judges

Score every trace with built-in judges, custom `Guidelines`, and deterministic checks.

```python
eval_results = mlflow.genai.evaluate(
    data=traces_df,
    scorers=[
        relevance,        # RelevanceToQuery — response answers the question
        safety,           # Safety — no harmful content
        groundedness,     # RetrievalGroundedness — grounded in retrieved data
        response_quality, # Guidelines — data-driven, direct answers
        sql_quality,      # Guidelines — sensible aggregations, WHERE, LIMIT
        has_response,     # code scorer — a text response exists
        no_error,         # code scorer — no error occurred
    ],
)
```

Each scorer is logged as a `yes`/`no` assessment column on the trace, so you can see
which conversations failed and why.

### 3. Generate space improvements

Collect the failed traces, pull the current space configuration, and have an LLM
propose copy-paste-ready fixes (text instructions, SQL expressions, example queries,
benchmarks). Review the recommendations and apply them to your Genie space, then
re-run stages 1–2 to measure the impact.

## Concepts

- **Genie space** — wraps Unity Catalog tables, instructions, SQL expressions, and
  benchmarks to guide natural-language-to-SQL translation.
- **Trace** — one Genie interaction (question → SQL → answer) captured for inspection.
- **Scorer / judge** — built-in, custom, or code-based check that grades a trace.

## References

- [Cookbook: Genie + MLflow](https://mlflow.org/cookbook/databricks-genie/)
- [Conversation tracing pipeline](https://mlflow.org/cookbook/genie-tracing-pipeline/)
- [Evaluation with LLM judges](https://mlflow.org/cookbook/genie-evaluation-judges/)
- [Space improvement generator](https://mlflow.org/cookbook/genie-space-analyzer/)
