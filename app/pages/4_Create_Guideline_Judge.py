"""Page 4 — Create Guideline Judge.

Define custom MLflow Guidelines judges (or quick-add a preset).
Judges are stored in st.session_state and shared with the Evaluate page's
Guidelines section for this session.
"""

import streamlit as st

from common import get_guideline_judges
from guideline_presets import GUIDELINE_JUDGES

st.title("🧭 Create Guideline Judge")
st.caption(
    "Define a custom MLflow Guidelines judge, or quick-add one of the "
    "presets. Judges here appear in the **Guidelines judges** section on the "
    "Evaluate page for this session."
)

judges = get_guideline_judges()

# ---------------------------------------------------------------------------
# Quick add from presets
# ---------------------------------------------------------------------------
st.subheader("Quick add from presets")
available_presets = [n for n in GUIDELINE_JUDGES if n not in judges]
if available_presets:
    preset_name = st.selectbox("Preset", available_presets)
    for g in GUIDELINE_JUDGES[preset_name]:
        st.caption(f"• {g}")
    if st.button("Add preset"):
        judges[preset_name] = list(GUIDELINE_JUDGES[preset_name])
        st.rerun()
else:
    st.caption("All presets have already been added.")

# ---------------------------------------------------------------------------
# Create a custom judge
# ---------------------------------------------------------------------------
st.subheader("Or create a custom judge")
with st.form("new_guideline_judge", clear_on_submit=True):
    name = st.text_input("Judge name", placeholder="e.g. mentions_time_period")
    guideline = st.text_area(
        "Guideline text (one rule per line)",
        placeholder="The response MUST ...",
        height=140,
        help="Each non-empty line becomes one guideline for this judge.",
    )
    submitted = st.form_submit_button("Add guideline judge", type="primary")

if submitted:
    lines = [ln.strip() for ln in guideline.splitlines() if ln.strip()]
    if not name.strip() or not lines:
        st.error("Both a name and at least one guideline line are required.")
    elif name.strip() in judges:
        st.error(f"A judge named '{name.strip()}' already exists.")
    else:
        judges[name.strip()] = lines
        st.success(f"Added guideline judge '{name.strip()}'.")

# ---------------------------------------------------------------------------
# Active judges
# ---------------------------------------------------------------------------
st.subheader("Active guideline judges")
if not judges:
    st.info("No guideline judges yet — add one above.")
else:
    for judge_name, guidelines in list(judges.items()):
        col1, col2 = st.columns([5, 1])
        with col1:
            st.markdown(f"**{judge_name}**")
            for g in guidelines:
                st.caption(f"• {g}")
        with col2:
            if st.button("Remove", key=f"remove_{judge_name}"):
                del judges[judge_name]
                st.rerun()
