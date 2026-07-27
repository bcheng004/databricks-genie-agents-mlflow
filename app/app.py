"""Navigation for the Genie MLflow app.

The UC-managed MLflow experiment is provisioned by the `uv` quickstart, and the
app reads its configuration (experiment name, SQL warehouse, Genie Space ID)
from environment variables set in `app.yaml`. Use the sidebar to trace Genie
conversations, evaluate them with LLM judges, and generate improvement
suggestions.
"""

import streamlit as st

st.navigation(
    [
        st.Page(
            "pages/1_Trace_Genie_Conversations.py",
            title="Trace Conversations",
            icon="💬",
            default=True,
        ),
        st.Page(
            "pages/4_Create_Guideline_Judge.py",
            title="Create Guideline Judge",
            icon="🧭",
        ),
        st.Page(
            "pages/2_Evaluate_With_Judges.py",
            title="Evaluate With Judges",
            icon="⚖️",
        ),
        st.Page(
            "pages/3_Improve_Genie_Space.py",
            title="Improve Genie Agents",
            icon="🛠️",
        ),
    ]
).run()
