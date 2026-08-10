"""Query the deployed web-search agent endpoint directly (no UC function).

Calls the Model Serving agent endpoint from your local process (where
credentials exist) via the OpenAI-compatible Responses API. Useful for testing
the endpoint the ``web_search`` UC tool delegates to.

    uv run query-web-search-agent "what is the latest Databricks Runtime LTS?"

Or non-interactively pick the profile/endpoint:

    uv run query-web-search-agent --profile mlflow-workshop \
        --endpoint web_search_agent "your question"
"""

from __future__ import annotations

import argparse
import sys

from ._common import (
    get_env_value,
    resolve_profile,
    validate_profile,
)
from .web_search_tool import DEFAULT_AGENT_ENDPOINT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="*", help="The web search question to ask.")
    parser.add_argument("--profile", default=None, help="Databricks CLI profile.")
    parser.add_argument(
        "--endpoint",
        default=None,
        help="Agent endpoint name (default: cached WEB_SEARCH_AGENT_ENDPOINT).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    profile = resolve_profile(args.profile)
    if not validate_profile(profile):
        sys.exit(
            f"Profile '{profile}' is not authenticated. Run:\n"
            f"  databricks auth login --profile {profile}"
        )

    endpoint = (
        args.endpoint
        or get_env_value("WEB_SEARCH_AGENT_ENDPOINT")
        or DEFAULT_AGENT_ENDPOINT
    )
    query = " ".join(args.query).strip() or input("Web search query: ").strip()
    if not query:
        sys.exit("A query is required.")

    from databricks.sdk import WorkspaceClient
    from databricks_openai import DatabricksOpenAI

    # DatabricksOpenAI must be built from a profiled WorkspaceClient, or it falls
    # back to the DEFAULT profile's host.
    w = WorkspaceClient(profile=profile)
    client = DatabricksOpenAI(workspace_client=w)

    print(f"Querying agent endpoint '{endpoint}' …")
    response = client.responses.create(
        model=endpoint,
        input=[{"role": "user", "content": query}],
    )
    print("\n" + (getattr(response, "output_text", "") or "(no output)"))


if __name__ == "__main__":
    main()
