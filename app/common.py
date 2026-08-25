"""Shared helpers used by every page of the Genie MLflow app."""

import logging
import os
from typing import List

import mlflow
import streamlit as st
from databricks.sdk import WorkspaceClient
from databricks_openai import DatabricksOpenAI

logger = logging.getLogger(__name__)

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
    "databricks-gpt-5-6-luna",
    "databricks-claude-sonnet-5",
    "databricks-claude-sonnet-4-6",
    "databricks-qwen35-122b-a10b",
    "databricks-gpt-5-5",
]


def get_obo_token() -> str | None:
    """Return the OBO access token forwarded by Databricks Apps, or None when running locally.

    Streamlit's header map is case-insensitive, but we look the header up under both
    casings defensively in case that behavior ever changes.
    """
    try:
        headers = st.context.headers
    except Exception:
        return None
    if not headers:
        return None
    for key in ("X-Forwarded-Access-Token", "x-forwarded-access-token"):
        token = headers.get(key)
        if token:
            return token
    return None


def _resolve_workspace_host() -> str:
    """Workspace URL used to construct an OBO client.

    ``DATABRICKS_HOST`` is auto-injected by the Databricks Apps runtime;
    ``DATABRICKS_WORKSPACE_URL`` is set explicitly in app.yaml as a backstop.
    """
    return (
        os.environ.get("DATABRICKS_HOST")
        or os.environ.get("DATABRICKS_WORKSPACE_URL")
        or ""
    ).strip()


def _in_databricks_apps() -> bool:
    """True when running inside the Databricks Apps runtime (SP creds injected)."""
    return bool(os.environ.get("DATABRICKS_APP_NAME"))


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

    In the Apps runtime the forwarded on-behalf-of-user token is required: falling back
    to the app's service principal authenticates as a principal that is NOT a member of
    the ``users`` group Foundation Model endpoints are granted to, so
    ``serving_endpoints.list()`` returns nothing and callers silently degrade to the
    fallback model list. Raise instead, so the misconfiguration surfaces in the logs
    rather than showing wrong data. Strips SP OAuth env vars before construction so the
    SDK does not see multiple auth methods simultaneously. Locally (no forwarded header)
    fall back to default credential resolution (profile/env).
    """
    token = get_obo_token()
    host = _resolve_workspace_host()
    if token and host:
        removed = _clean_env_for_obo()
        try:
            return WorkspaceClient(host=host, token=token)
        finally:
            os.environ.update(removed)
    if _in_databricks_apps():
        raise RuntimeError(
            "No on-behalf-of-user token available in the Databricks Apps runtime "
            f"(token={'present' if token else 'missing'}, host={host or 'unset'!r}). "
            "The app must authenticate as the logged-in user; refusing to fall back to "
            "the service principal, which cannot list Foundation Model endpoints. "
            "Enable user authorization on the workspace and declare the required "
            "user_api_scopes (model-serving, genie, sql, postgres)."
        )
    return WorkspaceClient()


def get_openai_client(w: WorkspaceClient | None = None) -> DatabricksOpenAI:
    """Return a DatabricksOpenAI client authenticated as the logged-in user (OBO).

    DatabricksOpenAI authenticates from a WorkspaceClient (via its http_client),
    not an ``api_key`` — passing ``api_key`` collides with the value it sets
    internally. So we hand it the OBO-authenticated WorkspaceClient, which
    carries the logged-in user's credentials in Databricks Apps (and the local
    profile otherwise).
    """
    return DatabricksOpenAI(workspace_client=w or get_workspace_client())


def init_mlflow() -> None:
    """Set the MLflow tracking URI. Call once before any mlflow.set_experiment()."""
    mlflow.set_tracking_uri("databricks")


_CHAT_TASK = "llm/v1/chat"

# Prefix shown before external-model endpoints in model dropdowns. st.selectbox
# renders option text literally (no per-option markdown/color — verified), so an
# emoji marker is the way to flag externals visually.
EXTERNAL_MODEL_MARKER = "🌐"


def _is_external_chat_endpoint(endpoint) -> bool:
    """True if any served entity is an external model exposing a chat task.

    External-model endpoints (OpenAI, Anthropic, Bedrock, etc. proxied through
    Model Serving) leave the top-level ``endpoint.task`` unset and carry the task
    on each served entity's ``external_model.task``.
    """
    config = getattr(endpoint, "config", None)
    served_entities = getattr(config, "served_entities", None) or []
    for entity in served_entities:
        external = getattr(entity, "external_model", None)
        if external is not None and getattr(external, "task", None) == _CHAT_TASK:
            return True
    return False


def _is_chat_endpoint(endpoint) -> bool:
    """True if a serving endpoint exposes a chat interface.

    Native Foundation Model / provisioned-throughput endpoints report the task
    at the top level (``endpoint.task``); external-model endpoints report it on
    a served entity (see ``_is_external_chat_endpoint``). Accept either.
    """
    return endpoint.task == _CHAT_TASK or _is_external_chat_endpoint(endpoint)


@st.cache_data(ttl=600)
def _list_chat_endpoints(_token: str | None = None) -> tuple[List[str], frozenset[str]]:
    """Return (model_names, external_names) for chat-capable serving endpoints.

    Names are sorted newest-launched first. ``external_names`` is the subset
    backed by external providers. A single scan feeds both the option list and
    the external-vs-native marking, so callers share one cached API call.
    ``_token`` keys the cache per user when OBO is active.

    Falls back to ``FALLBACK_JUDGE_MODELS`` (all treated as native) on error or
    when the workspace exposes no chat endpoints.
    """
    try:
        w = get_workspace_client()
        endpoints = [e for e in w.serving_endpoints.list() if _is_chat_endpoint(e)]
        endpoints.sort(key=lambda e: e.creation_timestamp or 0, reverse=True)
        names = [e.name for e in endpoints]
        external = frozenset(
            e.name for e in endpoints if _is_external_chat_endpoint(e)
        )
        if not names:
            logger.warning(
                "serving_endpoints.list() returned no chat endpoints (identity may "
                "lack CAN_VIEW / not be in the users group); using fallback list."
            )
            return list(FALLBACK_JUDGE_MODELS), frozenset()
        return names, external
    except Exception:
        logger.exception("Failed to list chat serving endpoints; using fallback list.")
        return list(FALLBACK_JUDGE_MODELS), frozenset()


def list_judge_models(_token: str | None = None) -> List[str]:
    """List chat-capable serving endpoint names, newest first.

    Includes native Foundation Model API endpoints and external-model endpoints
    (external providers proxied through Databricks Model Serving).
    """
    names, _ = _list_chat_endpoints(_token)
    return names


def list_external_models(_token: str | None = None) -> frozenset[str]:
    """Return the subset of chat endpoint names backed by external providers."""
    _, external = _list_chat_endpoints(_token)
    return external


def format_model_option(name: str, external: frozenset[str] | None = None) -> str:
    """Format a model name for a selectbox, marking external models.

    Display-only: st.selectbox's ``format_func`` changes the shown label, not
    the returned value (verified), so the bare ``name`` is what callers get back.
    Pass the ``external`` set from ``list_external_models()``; when omitted it is
    fetched (cached).
    """
    if external is None:
        external = list_external_models()
    return f"{EXTERNAL_MODEL_MARKER} {name}" if name in external else name


def require_experiment() -> tuple[str, str]:
    """Return (experiment_name, experiment_id) or stop the page if not configured.

    The experiment is provisioned by the `uv` quickstart and its name is passed
    to the app via the MLFLOW_EXPERIMENT_NAME env var (set in app.yaml). The
    experiment ID is resolved by name.

    MLFLOW_TRACING_SQL_WAREHOUSE_ID is only relevant for UC-backed traces, whose
    trace tables are queried through a SQL warehouse. When it is set, we ensure a
    warehouse is resolved. When it is unset (workspace-backed traces — traces
    live in the MLflow backend), we deliberately do NOT auto-detect one: doing so
    would make trace queries run against a warehouse the app has no grant on
    (the sql-warehouse app resource is removed for workspace-backed traces),
    yielding a PermissionDenied on the SQL endpoint.
    """
    name = os.environ.get("MLFLOW_EXPERIMENT_NAME")
    if not name:
        st.error(
            "MLFLOW_EXPERIMENT_NAME is not set. Run the `uv` quickstart to "
            "provision the MLflow experiment, then set it in app.yaml."
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

    return name, experiment.experiment_id


def genie_agent_id_input() -> str:
    """Render a Genie Agent ID text input pre-filled from the GENIE_AGENT_ID env var."""
    return st.text_input(
        "Genie Agent ID",
        value=os.environ.get("GENIE_AGENT_ID", "") or os.environ.get("GENIE_SPACE_ID", ""),  # GENIE_SPACE_ID for back-compat
        placeholder="e.g. 01f0123456789abc",
        help="The Genie agent ID (visible in the Genie agent URL).",
    )


def get_guideline_judges() -> dict[str, list[str]]:
    """Return the session's Guidelines judges ({name: [guideline, ...]}).

    Seeded once from the notebook-02 presets so the Evaluate page shows the
    built-in Genie judges by default; the "Create Guideline Judge" page adds to
    or removes from this same dict, shared across pages via st.session_state.
    """
    from guideline_presets import GUIDELINE_JUDGES

    if "guideline_judges" not in st.session_state:
        st.session_state["guideline_judges"] = {
            name: list(guidelines) for name, guidelines in GUIDELINE_JUDGES.items()
        }
    return st.session_state["guideline_judges"]
