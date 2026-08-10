"""BM25 code search over a Lakebase Postgres project (notebook 04).

Backs the ``code_grounded`` scorer on the Evaluate page: retrieves code chunks
from the Lakebase `code-search` project with a BM25 keyword query, then an LLM
judges whether a Genie response is actually grounded in that code.

Connections are made as the *logged-in user* (OBO) so the caller's Postgres
grants apply. The ``WorkspaceClient`` must therefore be built on Streamlit's
main thread — ``mlflow.genai.evaluate()`` runs scorers in worker threads where
``st.context.headers`` is unavailable — which is why ``CodeSearcher`` takes a
client instead of creating one lazily.
"""

import os
import threading
import time

from databricks.sdk import WorkspaceClient

# Defaults match the Lakebase project indexed in notebook 04; override via env.
DEFAULT_PROJECT = "code-search"
DEFAULT_BRANCH = "production"
DEFAULT_ENDPOINT = "primary"
DEFAULT_DATABASE = "databricks_postgres"

# Postgres credentials are valid for an hour; refresh well before that.
_TOKEN_TTL_SECONDS = 30 * 60

# Cap on the retrieved code echoed into a Feedback rationale, so assessment
# payloads stay readable in the MLflow UI.
MAX_RATIONALE_SNIPPET_CHARS = 4000

NO_RESULTS = "No relevant code found."

_BM25_SQL = """
    SELECT c.content, f.path, r.name AS repo_name, r.default_branch,
           c.start_line, c.end_line,
           c.ts <@> to_bm25query(to_tsvector('english', %(query)s), 'ix_chunks_ts_bm25'::regclass) AS score
    FROM chunks c
    JOIN files f ON f.id = c.file_id
    JOIN repos r ON r.id = f.repo_id
    ORDER BY c.ts <@> to_bm25query(to_tsvector('english', %(query)s), 'ix_chunks_ts_bm25'::regclass)
    LIMIT %(top_k)s
"""


def endpoint_path() -> str:
    """Return the Lakebase endpoint resource path.

    Prefers ``LAKEBASE_ENDPOINT`` (injected by Databricks Apps when a
    ``postgres`` resource is attached) and otherwise assembles the path from the
    project/branch/endpoint env vars. ``generate_database_credential`` only
    accepts an *endpoint* path, so a branch- or database-shaped value is
    normalized up to its branch and pointed at the endpoint.
    """
    injected = os.environ.get("LAKEBASE_ENDPOINT")
    if injected:
        return _as_endpoint_path(injected)
    project = os.environ.get("LAKEBASE_PROJECT", DEFAULT_PROJECT)
    branch = os.environ.get("LAKEBASE_BRANCH", DEFAULT_BRANCH)
    endpoint = os.environ.get("LAKEBASE_ENDPOINT_NAME", DEFAULT_ENDPOINT)
    return f"projects/{project}/branches/{branch}/endpoints/{endpoint}"


def _as_endpoint_path(resource: str) -> str:
    """Coerce a Lakebase resource path into ``projects/…/branches/…/endpoints/…``."""
    resource = resource.strip().rstrip("/")
    if "/endpoints/" in resource:
        return resource
    parts = resource.split("/")
    if len(parts) >= 4 and parts[2] == "branches":
        branch = "/".join(parts[:4])
        endpoint = os.environ.get("LAKEBASE_ENDPOINT_NAME", DEFAULT_ENDPOINT)
        return f"{branch}/endpoints/{endpoint}"
    return resource


def database_name() -> str:
    return os.environ.get("PGDATABASE") or os.environ.get(
        "LAKEBASE_DATABASE", DEFAULT_DATABASE
    )


class CodeSearcher:
    """BM25 search against the Lakebase code index, as the logged-in user.

    Construct one on Streamlit's main thread and pass it to the scorer factory;
    ``search()`` is then safe to call from the evaluation worker threads. Host,
    username, and credential are resolved once and reused under a lock.
    """

    def __init__(self, client: WorkspaceClient):
        self._client = client
        self._endpoint = endpoint_path()
        self._dbname = database_name()
        self._lock = threading.Lock()
        self._host: str | None = None
        self._user: str | None = None
        self._token: str | None = None
        self._token_issued_at = 0.0

    # -- connection details -------------------------------------------------
    def _resolve_host_and_user(self) -> tuple[str, str]:
        if self._host is None:
            host = os.environ.get("PGHOST")
            if not host:
                endpoint = self._client.postgres.get_endpoint(name=self._endpoint)
                host = endpoint.status.hosts.host
            self._host = host
        if self._user is None:
            self._user = self._client.current_user.me().user_name
        return self._host, self._user

    def _fresh_token(self) -> str:
        if self._token is None or time.time() - self._token_issued_at > _TOKEN_TTL_SECONDS:
            self._token = self._client.postgres.generate_database_credential(
                endpoint=self._endpoint
            ).token
            self._token_issued_at = time.time()
        return self._token

    def connection_info(self) -> dict:
        """Return the resolved connection details (for display/diagnostics)."""
        with self._lock:
            host, user = self._resolve_host_and_user()
        return {
            "endpoint": self._endpoint,
            "host": host,
            "database": self._dbname,
            "user": user,
        }

    # -- search -------------------------------------------------------------
    def search_rows(self, query: str, top_k: int = 5) -> list[tuple]:
        """Run the BM25 query and return raw result rows."""
        import psycopg

        with self._lock:
            host, user = self._resolve_host_and_user()
            token = self._fresh_token()

        with psycopg.connect(
            host=host,
            dbname=self._dbname,
            user=user,
            password=token,
            sslmode="require",
        ) as conn, conn.cursor() as cur:
            cur.execute(_BM25_SQL, {"query": query, "top_k": top_k})
            return cur.fetchall()

    def search(self, query: str, top_k: int = 5) -> str:
        """Return matching code chunks formatted for an LLM prompt."""
        rows = self.search_rows(query, top_k=top_k)
        return format_rows(rows) if rows else NO_RESULTS


def format_rows(rows: list[tuple]) -> str:
    """Format search rows into the snippet block passed to an LLM."""
    return "\n\n".join(
        f"--- {repo_name}: {path} (score: {score:.4f}) ---\n"
        f"{snippet_url(repo_name, branch, path, start_line, end_line)}\n{content}"
        for content, path, repo_name, branch, start_line, end_line, score in rows
    )


def snippet_url(repo_name: str, branch: str, path: str, start_line, end_line) -> str:
    """Return a GitHub permalink to the indexed chunk."""
    return (
        f"https://github.com/{repo_name}/blob/{branch}/{path}#L{start_line}-L{end_line}"
    )


def source_links(rows: list[tuple]) -> list[str]:
    """Return markdown links for the rows returned by ``search_rows``."""
    return [
        f"- [{repo_name}: {path} (L{start_line}-L{end_line})]"
        f"({snippet_url(repo_name, branch, path, start_line, end_line)}) — score {score:.4f}"
        for _content, path, repo_name, branch, start_line, end_line, score in rows
    ]
