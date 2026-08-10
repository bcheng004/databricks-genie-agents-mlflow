



# Databricks Genie Agents

Evaluate and improve [Databricks Genie](https://docs.databricks.com/genie/) agents
(text-to-SQL assistants) using MLflow. Genie conversations become MLflow traces you
can inspect, score with LLM judges, and turn into concrete configuration fixes — all
from a Streamlit app deployed as a [Databricks App](https://docs.databricks.com/dev-tools/databricks-apps/).

Based on the [MLflow Databricks Genie cookbook](https://mlflow.org/cookbook/databricks-genie/).

## Demo

<video src="docs/demo.mp4" controls width="100%"></video>

▶ [Watch the demo](docs/demo.mp4) — a walkthrough of tracing, evaluating, and improving a Genie agent.

## What you get

A four-page Streamlit app:

1. **Chat & Trace Genie** — chat with the Genie agent in an embedded view, then log its conversations as MLflow traces (question, generated SQL, executed query result, and text response).
2. **Create Guideline Judge** — define custom `Guidelines` judges (or quick-add presets) used on the Evaluate page.
3. **Evaluate With Judges** — pick traces from a table and score them with built-in judges, Guidelines judges, and code-based scorers.
4. **Improve Genie Agents** — feed failed traces + the agent config to an LLM that proposes copy-paste-ready fixes.

The app reads its configuration (experiment, warehouse, Genie agent) from environment
variables in `app/app.yaml`. The `uv run quickstart` command provisions the UC-managed
MLflow experiment and writes those values for you.

## Prerequisites

- A Databricks workspace with an existing Genie agent that has conversations
- The [Databricks CLI](https://docs.databricks.com/dev-tools/cli/) (`databricks`) authenticated to that workspace
- [`uv`](https://docs.astral.sh/uv/) installed

```bash
# Authenticate the CLI (creates a profile you'll select in the quickstart)
databricks auth login --profile my-workspace
```

## Quickstart

### 1. Provision the experiment

Run the quickstart with no flags to be guided interactively:

```bash
uv run quickstart
```

It walks you through the following **terminal inputs** — press Enter to accept the
shown default (in parentheses), or type a value:

| Prompt | What to enter | Default |
| --- | --- | --- |
| **Select a profile** | The Databricks CLI profile to use. A numbered list of your configured profiles is shown — type the **number**, the **name**, or a **new name** to authenticate as. | your current profile |
| **Experiment path** | Workspace path for the MLflow experiment (created if missing). | `/Workspace/Shared/genie-eval-traces` |
| **Catalog** | Unity Catalog catalog for trace storage. | `main` |
| **Schema** | UC schema for trace storage (created if missing). | `genie_traces` |
| **Table prefix** | Prefix for the trace tables. | `evals` |

The SQL warehouse is auto-detected (a Serverless Starter Warehouse if available). The
resolved config is written to `.env`, `app/app.yaml`, and the `databricks.yml` bundle
variables.

Prefer non-interactive (e.g. CI)? Pass any subset of flags to skip those prompts:

```bash
uv run quickstart \
  --profile my-workspace \
  --experiment-name /Workspace/Shared/genie-eval-traces \
  --catalog main --schema genie_traces --table-prefix evals \
  --warehouse-id abc123def456
```

Reruns are idempotent — an experiment already at that path is reused. Pass `--force`
to recreate it.

### 2. Attach a Genie Agent

```bash
uv run add-genie-agent
```

**Terminal inputs:**

| Prompt | What to enter |
| --- | --- |
| **Select a profile** | Same profile picker as the quickstart. |
| **Attach vs. create** | `1` to attach an existing agent by ID, or `2` to reuse/create one by title. |
| **Genie agent ID** | (option 1) The agent ID from the Genie agent URL, e.g. `01f0123456789abc`. |
| **Agent title** | (option 2) Reuses an agent with that title if found, otherwise creates a new one. |
| **Warehouse ID / tables / description** | (only when creating a new agent) SQL warehouse, one fully-qualified `catalog.schema.table` per line (blank to finish), and an optional description. |

Or skip the prompts with flags:

```bash
# Attach an existing agent you already have
uv run add-genie-agent --agent-id 01f0123456789abc

# Reuse by title, or create it from tables if not found
uv run add-genie-agent --title "Sales Genie" --warehouse-id abc123 \
  --table main.sales.orders --table main.sales.customers
```

This writes `GENIE_AGENT_ID` into `.env`, `app/app.yaml`, and the bundle variables.

### 3. Deploy the app

```bash
databricks bundle deploy -t dev
databricks bundle run genie_mlflow -t dev
```

The `run` command prints the app URL. Open it, then work through the sidebar pages:
Chat & Trace Genie → (optionally Create Guideline Judge) → Evaluate With Judges →
Improve Genie Agents. Re-run tracing and evaluation after applying fixes to measure impact.

> The app authenticates as the logged-in user (on-behalf-of). If you add or change
> the app's OAuth scopes, existing users must re-consent (sign out and back in, or
> open the app in a fresh session) to get a token with the new scopes.

## Configuration reference

`app/app.yaml` environment variables the app reads at runtime (populated by the CLIs):

| Variable | Set by | Purpose |
| --- | --- | --- |
| `MLFLOW_EXPERIMENT_NAME` | `quickstart` | Experiment the app reads/writes traces from. |
| `MLFLOW_TRACING_SQL_WAREHOUSE_ID` | `quickstart` | Warehouse for MLflow trace queries. |
| `DATABRICKS_WORKSPACE_URL` | `quickstart` | Workspace URL used to build the embedded Genie chat URL. |
| `DATABRICKS_WORKSPACE_ID` | `quickstart` | Workspace (org) ID used as the `o=` param in the embedded chat URL. |
| `GENIE_AGENT_ID` | `add-genie-agent` | Default Genie agent (overridable per page). |
| `ANALYZER_MODEL` | manual | Model for the Improve page's analyzer (default `databricks-claude-sonnet-5`). |

## Concepts

- **Genie agent** — wraps Unity Catalog tables, instructions, SQL expressions, and
  benchmarks to guide natural-language-to-SQL translation.
- **Trace** — one Genie interaction (question → SQL → answer) captured for inspection.
- **Scorer / judge** — built-in, custom `Guidelines`, or code-based check that grades a trace.

## References

- [Cookbook: Genie + MLflow](https://mlflow.org/cookbook/databricks-genie/)
- [Conversation tracing pipeline](https://mlflow.org/cookbook/genie-tracing-pipeline/)
- [Evaluation with LLM judges](https://mlflow.org/cookbook/genie-evaluation-judges/)
- [Agent improvement generator](https://mlflow.org/cookbook/genie-space-analyzer/)
