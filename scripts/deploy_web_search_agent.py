"""Log, register, and deploy the web-search agent to a Model Serving endpoint.

The agent (``scripts/web_search_agent.py``) is logged with MLflow "models from
code", registered into Unity Catalog, and deployed via ``databricks.agents.deploy``.
The resulting Model Serving endpoint is what the ``web_search`` UC tool calls via
``ai_query`` (and what ``uv run query-web-search-agent`` calls directly).

    uv run deploy-web-search-agent

Or non-interactively:

    uv run deploy-web-search-agent --profile mlflow-workshop \
        --catalog main --schema genie_traces --model-name web_search_agent \
        --experiment-name /Workspace/Shared/web-search-agent

The deploy run is logged to a dedicated MLflow experiment (created if missing;
defaults to /Workspace/Shared/web-search-agent). Writes
``WEB_SEARCH_AGENT_ENDPOINT`` (and the UC model name / experiment) to .env so the
tool registration step points at what was deployed.

NOTE: this DEPLOYS a serving endpoint (billable, several minutes to provision).
It only runs when you invoke it explicitly.
"""

from __future__ import annotations

import argparse
import os
import sys

from ._common import (
    ensure_env_file,
    prompt_value,
    resolve_profile,
    update_env_file,
    validate_profile,
)
from .web_search_agent import LLM_ENDPOINT

# The agent module logged as the model artifact (models-from-code).
_AGENT_SOURCE = "scripts/web_search_agent.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default=None, help="Databricks CLI profile.")
    parser.add_argument("--catalog", default=None, help="UC catalog for the model.")
    parser.add_argument("--schema", default=None, help="UC schema for the model.")
    parser.add_argument(
        "--experiment-name",
        default=None,
        help="MLflow experiment path for the deploy run (created if missing). "
        "Prompted if omitted; defaults to /Workspace/Shared/web-search-agent.",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="UC model name / endpoint base name (default: web_search_agent).",
    )
    parser.add_argument(
        "--scale-to-zero",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Scale the endpoint to zero when idle (default: on). Use "
        "--no-scale-to-zero to keep a replica always warm (no cold starts, "
        "but pays for idle compute).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_env_file()

    profile = resolve_profile(args.profile)
    if not validate_profile(profile):
        sys.exit(
            f"Profile '{profile}' is not authenticated. Run:\n"
            f"  databricks auth login --profile {profile}"
        )
    # MLflow runs a validation predict() after log_model, and the agent builds a
    # bare WorkspaceClient() — point ambient auth at the chosen profile so it
    # doesn't fall back to DEFAULT. (Same approach as quickstart.py.)
    os.environ["DATABRICKS_CONFIG_PROFILE"] = profile
    # Don't bundle uv.lock/pyproject.toml into the model artifact. This repo's
    # uv.lock resolves the whole app (streamlit, macOS wheels, …) against the
    # Databricks pypi-proxy registry; if serving restored from it via `uv sync`,
    # the build would try to rebuild that entire env in a Linux container behind
    # the proxy. Excluding it forces serving to use the clean pip_requirements
    # below. Must be set before `import mlflow`. See MLFLOW_LOG_UV_FILES.
    os.environ.setdefault("MLFLOW_LOG_UV_FILES", "false")

    catalog = prompt_value("Catalog", args.catalog, "UC_CATALOG", "main")
    schema = prompt_value("Schema", args.schema, "UC_SCHEMA", "genie_traces")
    model_name = prompt_value(
        "Model name", args.model_name, "WEB_SEARCH_MODEL_NAME", "web_search_agent"
    )
    experiment_name = prompt_value(
        "Experiment path",
        args.experiment_name,
        "WEB_SEARCH_EXPERIMENT_NAME",
        "/Workspace/Shared/web-search-agent",
    )
    uc_model = f"{catalog}.{schema}.{model_name}"

    import mlflow
    from databricks import agents
    from mlflow.models.resources import DatabricksServingEndpoint

    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")
    # Create-or-reuse the experiment so the deploy run always lands somewhere
    # deterministic rather than the ambient default experiment.
    experiment = mlflow.set_experiment(experiment_name)
    print(f"  Experiment: {experiment.name} ({experiment.experiment_id})")

    print(f"[1/3] Logging agent ({_AGENT_SOURCE}) to MLflow …")
    with mlflow.start_run():
        logged = mlflow.pyfunc.log_model(
            name=model_name,
            python_model=_AGENT_SOURCE,
            # Declares the FM endpoint the agent calls, so deployment provisions a
            # credential the served agent uses to reach it.
            resources=[DatabricksServingEndpoint(endpoint_name=LLM_ENDPOINT)],
            pip_requirements=["mlflow", "databricks-openai", "databricks-sdk"],
        )
    print(f"  Logged: {logged.model_uri}")

    print(f"[2/3] Registering into Unity Catalog as {uc_model} …")
    registered = mlflow.register_model(model_uri=logged.model_uri, name=uc_model)
    print(f"  Registered version {registered.version}")

    print("[3/3] Deploying to Model Serving (provisioning may take minutes) …")
    deployment = agents.deploy(
        uc_model,
        int(registered.version),
        scale_to_zero=args.scale_to_zero,
    )
    endpoint_name = deployment.endpoint_name
    print(f"  Endpoint: {endpoint_name}")
    print(f"  Query URL: {getattr(deployment, 'query_endpoint', '(see UI)')}")

    update_env_file("WEB_SEARCH_MODEL_NAME", model_name)
    update_env_file("WEB_SEARCH_EXPERIMENT_NAME", experiment_name)
    update_env_file("WEB_SEARCH_AGENT_ENDPOINT", endpoint_name)
    print(
        f"\nDone. Agent deployed as endpoint: {endpoint_name}\n"
        "Next:\n"
        "  uv run create-web-search-tool   # register the UC tool that ai_query's it\n"
        "  uv run query-web-search-agent 'your question'   # call it directly"
    )


if __name__ == "__main__":
    main()
