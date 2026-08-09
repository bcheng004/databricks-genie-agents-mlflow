"""Page 5 — Chat With Genie.

Embeds the Genie agent chat UI directly in the app via an iframe so users can
ask questions without leaving the app. The embed URL is built from the app's
configured host, workspace (org) ID, and Genie agent ID.
"""

import os

import streamlit as st
import streamlit.components.v1 as components

from common import genie_agent_id_input, get_workspace_client

EMBED_HEIGHT = 700

st.title("💬 Chat With Genie")
st.caption(
    "Chat with the Genie agent directly in the app. The embedded view uses your "
    "logged-in Databricks session for authentication."
)

agent_id = genie_agent_id_input()


def _workspace_host() -> str | None:
    """Return the workspace host (scheme + domain), without a trailing slash.

    Prefers DATABRICKS_WORKSPACE_URL written by `uv run quickstart`, then falls
    back to the platform-injected DATABRICKS_HOST.
    """
    host = (
        os.environ.get("DATABRICKS_WORKSPACE_URL")
        or os.environ.get("DATABRICKS_HOST")
        or ""
    ).strip()
    if not host:
        return None
    if not host.startswith("http"):
        host = f"https://{host}"
    return host.rstrip("/")


@st.cache_data(ttl=3600)
def _workspace_id() -> str | None:
    """Return the workspace (org) ID used as the `o=` query param in the embed URL.

    Prefers DATABRICKS_WORKSPACE_ID written by `uv run quickstart`, then falls
    back to resolving it at runtime via the WorkspaceClient.
    """
    from_env = os.environ.get("DATABRICKS_WORKSPACE_ID", "").strip()
    if from_env:
        return from_env
    try:
        return str(get_workspace_client().get_workspace_id())
    except Exception:
        return None


host = _workspace_host()
org_id = _workspace_id()

if not agent_id.strip():
    st.info("Enter a Genie Agent ID above to load the chat.")
    st.stop()

if not host:
    st.error("DATABRICKS_HOST is not set — cannot build the Genie embed URL.")
    st.stop()

embed_url = f"{host}/embed/genie/rooms/{agent_id.strip()}"
if org_id:
    embed_url += f"?o={org_id}"

with st.expander("Embed URL"):
    st.code(embed_url, language="text")

components.html(
    f"""
    <iframe
      src="{embed_url}"
      width="100%"
      height="{EMBED_HEIGHT}"
      frameborder="0"
      allow="clipboard-write">
    </iframe>
    """,
    height=EMBED_HEIGHT,
)

st.caption(
    "If the frame is blank, open the embed URL above in a new tab — some browsers "
    "or workspace settings block third-party iframe cookies."
)
