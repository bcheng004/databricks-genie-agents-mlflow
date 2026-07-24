"""Shared helpers used by every page of the Genie MLflow app."""

import os
from typing import List

import mlflow
import streamlit as st
from databricks.sdk import WorkspaceClient
from databricks_openai import DatabricksOpenAI

# Env vars the Databricks Apps runtime sets for the service principal (OAuth M2M).
# They must be absent when constructing an OBO client, otherwise the SDK raises
# "more than one authorization method configured: oauth and pat".
_SP_OAUTH_KEYS = (
    "DATABRICKS_CLIENT_ID",
    "DATABRICKS_CLIENT_SECRET",
    "DATABRICKS_TOKEN",
    "ARM_CLIENT_ID",
    "ARM_CLIENT_SECRET",
    "ARM_TENANT_ID",
)

FALLBACK_JUDGE_MODELS = [
    "databricks-claude-sonnet-5",
    "databricks-claude-sonnet-4-6",
    "databricks-qwen35-122b-a10b",
    "databricks-gpt-5-5",
]


def get_obo_token() -> str | None:
    """Return the OBO access token forwarded by Databricks Apps, or None when running locally."""
    try:
        return st.context.headers.get("X-Forwarded-Access-Token")
    except Exception:
        return None


def _clean_env_for_obo() -> dict:
    """Remove SP OAuth/PAT env vars and return them so the caller can restore."""
    removed = {}
    for key in _SP_OAUTH_KEYS:
        val = os.environ.pop(key, None)
        if val is not None:
            removed[key] = val
    return removed


def get_workspace_client() -> WorkspaceClient:
    """Return a WorkspaceClient authenticated as the logged-in user (OBO) in Databricks Apps.

    Strips SP OAuth env vars before construction so the SDK does not see multiple
    auth methods simultaneously. Falls back to default credential resolution locally.
    """
    token = get_obo_token()
    host = os.environ.get("DATABRICKS_HOST", "")
    if token and host:
        removed = _clean_env_for_obo()
        try:
            client = WorkspaceClient(host=host, token=token)
        finally:
            os.environ.update(removed)
        return client
    return WorkspaceClient()


def get_openai_client() -> DatabricksOpenAI:
    """Return a DatabricksOpenAI client authenticated as the logged-in user (OBO)."""
    token = get_obo_token()
    host = os.environ.get("DATABRICKS_HOST", "")
    if token and host:
        return DatabricksOpenAI(api_key=token, base_url=f"{host}/serving-endpoints")
    return DatabricksOpenAI()


def init_mlflow() -> None:
    """Set the MLflow tracking URI. Call once before any mlflow.set_experiment()."""
    mlflow.set_tracking_uri("databricks")


@st.cache_data(ttl=600)
def detect_warehouse_id(_token: str | None = None) -> str | None:
    """Return a SQL warehouse ID, setting MLFLOW_TRACING_SQL_WAREHOUSE_ID as a side effect.

    Priority: env var already set → Serverless Starter Warehouse → first warehouse found.
    _token is included so the cache is keyed per user when OBO is active.
    """
    existing = os.environ.get("MLFLOW_TRACING_SQL_WAREHOUSE_ID")
    if existing:
        return existing
    try:
        w = get_workspace_client()
        warehouses = list(w.warehouses.list())
        wh = next(
            (
                wh
                for wh in warehouses
                if "Serverless Starter Warehouse" in (wh.name or "")
                and wh.enable_serverless_compute
            ),
            warehouses[0] if warehouses else None,
        )
        if wh:
            os.environ["MLFLOW_TRACING_SQL_WAREHOUSE_ID"] = wh.id
            return wh.id
    except Exception:
        pass
    return None


@st.cache_data(ttl=600)
def list_judge_models(_token: str | None = None) -> List[str]:
    """List chat-capable Databricks Foundation Model API endpoints, newest first.

    Sorted by the endpoint's creation (launch) date, most recently launched
    first. _token is included so the cache is keyed per user when OBO is active.
    """
    try:
        w = get_workspace_client()
        endpoints = [e for e in w.serving_endpoints.list() if e.task == "llm/v1/chat"]
        endpoints.sort(key=lambda e: e.creation_timestamp or 0, reverse=True)
        return [e.name for e in endpoints] or FALLBACK_JUDGE_MODELS
    except Exception:
        return FALLBACK_JUDGE_MODELS


def require_experiment() -> tuple[str, str]:
    """Return (experiment_name, experiment_id) or stop the page if not configured.

    The experiment is provisioned by the `uv` quickstart and its name is passed
    to the app via the MLFLOW_EXPERIMENT_NAME env var (set in app.yaml). The
    experiment ID is resolved by name. MLFLOW_TRACING_SQL_WAREHOUSE_ID is also
    read from the environment so MLflow trace queries work on every page.
    """
    name = os.environ.get("MLFLOW_EXPERIMENT_NAME")
    if not name:
        st.error(
            "MLFLOW_EXPERIMENT_NAME is not set. Run the `uv` quickstart to "
            "provision the UC-managed MLflow experiment, then set it in app.yaml."
        )
        st.stop()

    init_mlflow()
    experiment = mlflow.get_experiment_by_name(name)
    if experiment is None:
        st.error(
            f"Experiment `{name}` was not found. Run the `uv` quickstart to "
            "provision it, or correct MLFLOW_EXPERIMENT_NAME in app.yaml."
        )
        st.stop()

    if not os.environ.get("MLFLOW_TRACING_SQL_WAREHOUSE_ID"):
        detect_warehouse_id(get_obo_token())
    return name, experiment.experiment_id


def genie_space_id_input() -> str:
    """Render a Genie Space ID text input pre-filled from the GENIE_SPACE_ID env var."""
    return st.text_input(
        "Genie Space ID",
        value=os.environ.get("GENIE_SPACE_ID", ""),
        placeholder="e.g. 01f0123456789abc",
        help="The Genie space ID (visible in the Genie space URL).",
    )
