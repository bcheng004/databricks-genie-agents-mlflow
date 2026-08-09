"""Web-search agent (MLflow ``ResponsesAgent``), deployed to Model Serving.

This is the agent that actually performs web search. It wraps a Databricks
foundation model that has the built-in ``web_search`` tool
(https://docs.databricks.com/aws/en/machine-learning/model-serving/web-search):
the model issues live queries and returns a grounded answer with source URLs, so
no third-party search key is needed.

Why an agent endpoint (and not a UC function doing the search directly)? A Unity
Catalog function runs in a sandbox with **no credentials and no network egress**,
so it cannot call a serving endpoint itself. A deployed agent *does* have ambient
credentials and can reach ``databricks-gpt-5``. The UC ``web_search`` tool
therefore delegates here via ``ai_query('<this-endpoint>', request)`` — see
``scripts/web_search_tool.py``.

Logged with MLflow "models from code": ``mlflow.models.set_model(AGENT)`` at
import time makes this module the model artifact. ``deploy_web_search_agent.py``
logs, registers, and deploys it.
"""

from __future__ import annotations

import mlflow
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentResponse

# Foundation model that runs the search. Must be a pay-per-token endpoint whose
# model supports the built-in web-search tool (OpenAI Responses-API models do).
LLM_ENDPOINT = "databricks-gpt-5"

mlflow.openai.autolog()


class WebSearchAgent(ResponsesAgent):
    """A minimal agent: forward the user's request to an LLM with web search on."""

    def _client(self):
        # Inside Model Serving the WorkspaceClient resolves ambient credentials;
        # DatabricksOpenAI wraps it with the OpenAI-compatible surface.
        from databricks_openai import DatabricksOpenAI

        return DatabricksOpenAI()

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        client = self._client()
        # request.input is a list of Responses-API items ({"role","content"}...).
        messages = [item.model_dump() for item in request.input]

        response = client.responses.create(
            model=LLM_ENDPOINT,
            input=messages,
            tools=[{"type": "web_search"}],
        )

        answer = (getattr(response, "output_text", "") or "").strip()

        # Attach any URL citations the model produced as annotations on the item.
        annotations: list[dict] = []
        for item in getattr(response, "output", None) or []:
            for part in getattr(item, "content", None) or []:
                for ann in getattr(part, "annotations", None) or []:
                    url = getattr(ann, "url", None)
                    if url:
                        annotations.append(
                            {"type": "url_citation", "url": url,
                             "title": getattr(ann, "title", "") or url}
                        )

        if not answer:
            answer = "No web results were found for that query."

        output_item = self.create_text_output_item(
            text=answer,
            id=getattr(response, "id", "") or "web_search",
            annotations=annotations or None,
        )
        return ResponsesAgentResponse(output=[output_item])


AGENT = WebSearchAgent()
mlflow.models.set_model(AGENT)
