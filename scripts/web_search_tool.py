"""The ``web_search`` custom agent tool, as a Unity Catalog **SQL** function.

Custom agent tool per
https://docs.databricks.com/aws/en/agents/custom-agents/create-custom-tool. The
tool is a UC SQL function that calls ``ai_query`` against the web-search agent
endpoint (``scripts/web_search_agent.py``, deployed by
``deploy_web_search_agent.py``).

Why SQL + ``ai_query`` rather than a Python UDF? A UC function runs sandboxed
with no credentials and no network egress, so a Python UDF calling
``WorkspaceClient()`` fails with ``cannot configure default credentials``.
``ai_query`` instead authenticates through the SQL **query execution context**
(the definer just needs ``CAN QUERY`` on the endpoint) and runs the search
server-side inside the agent — so the sandbox never needs creds or egress.

``ai_query`` cannot itself enable the web-search tool (its ``modelParameters``
allowlist rejects ``web_search``/``google_search`` and it has no ``tools`` arg),
which is exactly why the search lives in the agent endpoint and the tool only
passes a plain request string.
"""

from __future__ import annotations

# Default agent serving endpoint the tool queries. Overridden by the deploy step,
# which writes the resolved name so the SQL function points at what was deployed.
DEFAULT_AGENT_ENDPOINT = "web_search_agent"

# Unqualified function name; the registration CLI qualifies it with catalog.schema.
FUNCTION_NAME = "web_search"


def build_sql_function_body(catalog: str, schema: str, endpoint: str) -> str:
    """Return the ``CREATE FUNCTION`` SQL for the ``web_search`` tool.

    The function takes a natural-language ``query`` and returns the agent's
    grounded answer. ``ai_query`` sends the query as a chat request and returns
    the response text; the agent runs live web search server-side.

    Args:
        catalog: Unity Catalog catalog to create the function in.
        schema: Unity Catalog schema to create the function in.
        endpoint: Name of the deployed web-search agent serving endpoint.

    Returns:
        A ``CREATE OR REPLACE FUNCTION`` statement as a string.
    """
    fq_name = f"`{catalog}`.`{schema}`.{FUNCTION_NAME}"
    # The COMMENT is what the calling agent reads to decide when to use the tool,
    # so keep it action-oriented. Endpoint name is embedded as a SQL literal.
    return f"""CREATE OR REPLACE FUNCTION {fq_name}(
  query STRING COMMENT 'A specific, self-contained web search query or question.'
)
RETURNS STRING
COMMENT 'Search the public web and return a grounded answer with source URLs. Use for current, real-world, or post-training information: recent events, prices, releases, docs, or facts that may have changed.'
RETURN ai_query(
  '{endpoint}',
  CONCAT(
    'Search the web and answer concisely, citing source URLs. Question: ',
    query
  )
)"""
