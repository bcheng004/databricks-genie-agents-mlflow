"""Register the ``web_search`` custom agent tool as a Unity Catalog SQL function.

Follows https://docs.databricks.com/aws/en/agents/custom-agents/create-custom-tool.
The tool is a UC SQL function that calls ``ai_query`` against the web-search agent
endpoint (deploy it first with ``uv run deploy-web-search-agent``). ``ai_query``
authenticates via the SQL query context, so the function works despite the UC
sandbox having no credentials — see ``scripts/web_search_tool.py`` for the why.

Run with no flags to be prompted for the profile, catalog, and schema (defaults
come from the cached .env quickstart values):

    uv run create-web-search-tool

Or pass flags to skip the prompts (useful for CI):

    uv run create-web-search-tool --profile mlflow-workshop \
        --catalog main --schema genie_traces --endpoint web_search_agent

After registering, the script test-executes the function so you can see a real
grounded answer. Attach the tool to an agent by its fully-qualified name, e.g.:

    from databricks_langchain import UCFunctionToolkit
    tools = UCFunctionToolkit(function_names=["main.genie_traces.web_search"]).tools
"""

from __future__ import annotations

import argparse
import sys

from ._common import (
    ensure_env_file,
    get_env_value,
    prompt_value,
    resolve_profile,
    update_env_file,
    validate_profile,
)
from .web_search_tool import (
    DEFAULT_AGENT_ENDPOINT,
    FUNCTION_NAME,
    build_sql_function_body,
)

# A harmless smoke-test query run after registration to prove the tool works.
_TEST_QUERY = "What is the latest long-term support (LTS) Databricks Runtime version?"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        default=None,
        help="Databricks CLI profile. Prompted for interactively if omitted.",
    )
    parser.add_argument(
        "--catalog",
        default=None,
        help="Unity Catalog catalog to register the function in. Prompted if omitted.",
    )
    parser.add_argument(
        "--schema",
        default=None,
        help="Unity Catalog schema to register the function in. Prompted if omitted.",
    )
    parser.add_argument(
        "--endpoint",
        default=None,
        help="Web-search agent serving endpoint name (default: cached "
        "WEB_SEARCH_AGENT_ENDPOINT, else 'web_search_agent').",
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Register the function without running the post-registration smoke test.",
    )
    return parser.parse_args()


def check_endpoint(w, endpoint: str) -> None:
    """Warn (don't abort) if the agent endpoint isn't deployed/ready yet."""
    try:
        ep = w.serving_endpoints.get(endpoint)
        ready = getattr(getattr(ep, "state", None), "ready", None)
        print(f"  Agent endpoint '{endpoint}' state: {ready}")
    except Exception as exc:  # noqa: BLE001 — advisory only
        print(
            f"  (warning) serving endpoint '{endpoint}' not found: {exc}\n"
            f"           Deploy it first: uv run deploy-web-search-agent\n"
            f"           (ai_query needs a Model Serving endpoint to call.)"
        )


def main() -> None:
    args = parse_args()
    ensure_env_file()

    profile = resolve_profile(args.profile)
    if not validate_profile(profile):
        sys.exit(
            f"Profile '{profile}' is not authenticated. Run:\n"
            f"  databricks auth login --profile {profile}"
        )

    catalog = prompt_value("Catalog", args.catalog, "UC_CATALOG", "main")
    schema = prompt_value("Schema", args.schema, "UC_SCHEMA", "genie_traces")
    endpoint = (
        args.endpoint
        or get_env_value("WEB_SEARCH_AGENT_ENDPOINT")
        or DEFAULT_AGENT_ENDPOINT
    )

    from databricks.sdk import WorkspaceClient
    from unitycatalog.ai.core.databricks import DatabricksFunctionClient

    w = WorkspaceClient(profile=profile)
    print(f"[1/3] Checking web-search agent endpoint '{endpoint}' …")
    check_endpoint(w, endpoint)

    print(f"[2/3] Registering {FUNCTION_NAME} into {catalog}.{schema} …")
    client = DatabricksFunctionClient(client=w)
    sql_body = build_sql_function_body(catalog, schema, endpoint)
    function_info = client.create_function(sql_function_body=sql_body)
    func_name = f"{catalog}.{schema}.{FUNCTION_NAME}"
    print(f"  Registered {getattr(function_info, 'full_name', None) or func_name}")

    if not args.skip_test:
        print(f"[3/3] Smoke-testing: {_TEST_QUERY!r} …")
        try:
            result = client.execute_function(
                function_name=func_name,
                parameters={"query": _TEST_QUERY},
            )
            # execute_function returns a result whose `error` holds any failure
            # message (value is None on error) — surface it instead of a bare None.
            error = getattr(result, "error", None)
            if error:
                print(f"  (warning) tool raised inside the sandbox:\n    {error}")
            else:
                print("  ---- tool output ----")
                print(getattr(result, "value", result))
                print("  ---------------------")
        except Exception as exc:  # noqa: BLE001 — registration already succeeded
            print(f"  (warning) smoke test failed: {exc}")
    else:
        print("[3/3] Skipping smoke test (--skip-test).")

    update_env_file("WEB_SEARCH_TOOL_NAME", func_name)
    print(
        f"\nDone. Tool registered as: {func_name}\n"
        "Attach it to an agent, e.g.:\n"
        "  from databricks_langchain import UCFunctionToolkit\n"
        f'  tools = UCFunctionToolkit(function_names=["{func_name}"]).tools'
    )


if __name__ == "__main__":
    main()
