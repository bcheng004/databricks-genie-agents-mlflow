"""Page 1 — Chat & Trace Genie.

Two halves of one workflow on a single page:

  * **Chat** embeds the Genie agent so you can ask questions in-app. The embed
    URL is built from the workspace host, workspace (org) ID, and agent ID.
  * **Trace** pulls the resulting conversations from the Genie Space and logs
    each new message as an MLflow trace (question, generated SQL, executed
    query result, and text response). Re-run safely — traced messages skip.
"""

import os

import mlflow
import streamlit as st
import streamlit.components.v1 as components

from common import genie_agent_id_input, get_workspace_client, require_experiment

# Height of the embedded Genie chat iframe.
EMBED_HEIGHT = 700
# Cap logged result rows so traces stay lightweight for the evaluation pages.
MAX_RESULT_ROWS = 100

st.title("💬 Chat & Trace Genie")
st.caption(
    "Chat with the Genie agent in the app, then log those conversations as "
    "MLflow traces for evaluation. Both tabs use the same Genie Agent ID below."
)

agent_id = genie_agent_id_input()


# ---------------------------------------------------------------------------
# Chat embed helpers
# ---------------------------------------------------------------------------
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


def render_chat(agent_id: str) -> None:
    """Render the embedded Genie chat for the given agent."""
    if not agent_id.strip():
        st.info("Enter a Genie Agent ID above to load the chat.")
        return

    host = _workspace_host()
    if not host:
        st.error("DATABRICKS_WORKSPACE_URL / DATABRICKS_HOST is not set — cannot build the embed URL.")
        return

    embed_url = f"{host}/embed/genie/rooms/{agent_id.strip()}"
    org_id = _workspace_id()
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
        "If the frame is blank, open the embed URL above in a new tab — some "
        "browsers or workspace settings block third-party iframe cookies."
    )


# ---------------------------------------------------------------------------
# Trace helpers
# ---------------------------------------------------------------------------
def run_trace(space_id: str, experiment_name: str, experiment_id: str) -> None:
    """Pull Genie conversations and log each new message as an MLflow trace."""
    w = get_workspace_client()
    mlflow.set_experiment(experiment_name)

    with st.spinner("Fetching existing traces…"):
        existing_traces = mlflow.search_traces(
            locations=[experiment_id], return_type="list"
        )
        already_traced = {
            t.info.tags.get("message_id")
            for t in existing_traces
            if t.info.tags.get("message_id")
        }
    st.write(f"Found **{len(already_traced)}** already-traced message(s) — will skip.")

    with st.spinner("Fetching Genie conversations…"):
        conversations = w.genie.list_conversations(space_id=space_id, include_all=True)

    traced = 0
    errors = 0
    progress = st.progress(0.0, text="Logging traces…")
    convos = list(conversations.conversations or [])

    for i, convo in enumerate(convos):
        try:
            messages = w.genie.list_conversation_messages(
                space_id=space_id, conversation_id=convo.conversation_id
            )
        except Exception as exc:
            st.warning(f"Could not fetch messages for conversation {convo.conversation_id}: {exc}")
            errors += 1
            continue

        for msg in messages.messages or []:
            if not msg.content:
                continue
            if msg.message_id in already_traced:
                continue

            attachments = msg.attachments or []
            sql_att = next((a for a in attachments if a.query), None)
            text_att = next((a for a in attachments if a.text), None)

            # The message attachment carries only the generated SQL; the executed
            # result set is fetched separately by attachment id.
            query_result = None
            if sql_att:
                try:
                    res = w.genie.get_message_attachment_query_result(
                        space_id=space_id,
                        conversation_id=convo.conversation_id,
                        message_id=msg.message_id,
                        attachment_id=sql_att.attachment_id,
                    )
                    sr = res.statement_response
                    if sr and sr.result and sr.manifest:
                        cols = [c.name for c in sr.manifest.schema.columns]
                        rows = sr.result.data_array or []
                        query_result = {
                            "columns": cols,
                            "rows": rows[:MAX_RESULT_ROWS],
                            "row_count": len(rows),
                            "truncated": len(rows) > MAX_RESULT_ROWS,
                        }
                except Exception as exc:
                    query_result = {"error": str(exc)}

            with mlflow.start_span(name="genie_interaction") as span:
                span.set_inputs({"question": msg.content})
                span.set_outputs(
                    {
                        "response": (text_att.text.content if text_att else None),
                        "generated_sql": (sql_att.query.query if sql_att else None),
                        "query_result": query_result,
                        "error": str(msg.error) if msg.error else None,
                    }
                )
                mlflow.update_current_trace(tags={"message_id": msg.message_id})

            traced += 1

        progress.progress((i + 1) / max(len(convos), 1), text=f"Processed {i + 1}/{len(convos)} conversations…")

    progress.empty()

    if errors:
        st.warning(f"{errors} conversation(s) could not be fetched — check permissions.")

    if traced == 0:
        st.success("No new messages to trace — everything is already up to date.")
    else:
        st.success(f"Logged **{traced}** new trace(s) to experiment `{experiment_name}`.")


def render_trace(agent_id: str) -> None:
    """Render the trace controls. The experiment check only gates this tab."""
    # require_experiment() stops the run if the experiment isn't configured;
    # the Chat tab is rendered before this, so its content is already shown.
    experiment_name, experiment_id = require_experiment()
    st.info(f"Experiment: `{experiment_name}`")
    st.caption(
        "Pulls conversations from the Genie Space and logs each new message as "
        "an MLflow trace. Re-run safely — messages already traced are skipped."
    )

    if st.button("Trace new conversations", type="primary"):
        if not agent_id.strip():
            st.error("Please enter a Genie Agent ID above.")
        else:
            run_trace(agent_id.strip(), experiment_name, experiment_id)


# ---------------------------------------------------------------------------
# Layout — Chat tab renders first so it survives the Trace tab's experiment gate
# ---------------------------------------------------------------------------
tab_chat, tab_trace = st.tabs(["💬 Chat", "🔎 Trace"])

with tab_chat:
    render_chat(agent_id)

with tab_trace:
    render_trace(agent_id)
