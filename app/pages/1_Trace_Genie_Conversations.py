"""Page 1 — Trace Genie Conversations (notebook 01).

Pulls every conversation from a Genie Space, skips messages already traced,
and logs each new message as an MLflow trace with the question, SQL, executed
query result, and text response.
"""

import mlflow
import streamlit as st

from common import genie_agent_id_input, get_workspace_client, require_experiment

# Cap logged result rows so traces stay lightweight for the evaluation pages.
MAX_RESULT_ROWS = 100

st.title("💬 Trace Genie Conversations")
st.caption(
    "Pulls conversations from a Genie Space and logs each new message as an "
    "MLflow trace. Re-run safely — messages already traced are skipped."
)

experiment_name, experiment_id = require_experiment()
st.info(f"Experiment: `{experiment_name}`")

space_id = genie_agent_id_input()

if not st.button("Trace new conversations", type="primary"):
    st.stop()

if not space_id.strip():
    st.error("Please enter a Genie Space ID.")
    st.stop()

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
    conversations = w.genie.list_conversations(space_id=space_id.strip(), include_all=True)

traced = 0
errors = 0
progress = st.progress(0.0, text="Logging traces…")
convos = list(conversations.conversations or [])

for i, convo in enumerate(convos):
    try:
        messages = w.genie.list_conversation_messages(
            space_id=space_id.strip(), conversation_id=convo.conversation_id
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
                    space_id=space_id.strip(),
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
