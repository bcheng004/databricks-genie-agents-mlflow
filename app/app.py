"""Navigation for the Genie MLflow app.

The UC-managed MLflow experiment is provisioned by the `uv` quickstart, and the
app reads its configuration (experiment name, SQL warehouse, Genie Space ID)
from environment variables set in `app.yaml`. Use the sidebar to trace Genie
conversations, evaluate them with LLM judges, and generate improvement
suggestions.
"""

import base64
import os

import streamlit as st

_HERE = os.path.dirname(__file__)


def _img_tag(path: str, mime: str, width: int) -> str:
    data = base64.b64encode(open(path, "rb").read()).decode()
    return (
        f'<img src="data:{mime};base64,{data}" '
        f'width="{width}" style="display:block;margin:0 auto;" />'
    )


with st.sidebar:
    st.html(
        '<div style="text-align:center;padding:8px 0 4px">'
        + _img_tag(os.path.join(_HERE, "static", "databricks-logo.png"), "image/png", 180)
        + "<br/>"
        + _img_tag(os.path.join(_HERE, "static", "genie-logo.png"), "image/png", 64)
        + "<br/>"
        + _img_tag(os.path.join(_HERE, "static", "mlflow-logo.svg"), "image/svg+xml", 110)
        + "</div>"
    )
    st.divider()

st.navigation(
    [
        st.Page(
            "pages/1_Trace_Genie_Conversations.py",
            title="Trace Conversations",
            icon="💬",
            default=True,
        ),
        st.Page(
            "pages/5_Chat_With_Genie.py",
            title="Chat With Genie",
            icon="🗨️",
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
