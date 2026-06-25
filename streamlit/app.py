# app.py — Streamlit frontend stub
#
# Phase 0 purpose: prove the container starts and serves a page.
# All real UI will be built in later phases.

import os
import streamlit as st

# os.getenv reads an environment variable injected by Docker Compose.
# The second argument ("development") is the fallback value used if
# the variable is not set — prevents a crash in case .env is missing.
app_env = os.getenv("APP_ENV", "development")

# st.set_page_config must be the first Streamlit call in the script.
# Any st.* call before it raises a StreamlitAPIException.
# page_title sets the browser tab title.
# layout="wide" uses the full browser width instead of a narrow centered column.
st.set_page_config(
    page_title="Intelligent Document Processing Platform",
    layout="wide"
)

# st.title renders an H1 heading.
st.title("Intelligent Document Processing Platform")

# st.caption renders smaller muted text — appropriate for a subtitle.
st.caption("Phase 0 — Container scaffold. No application logic yet.")

# st.divider draws a horizontal rule. Purely cosmetic.
st.divider()

# st.success renders a green status box. We use it here to make it
# visually obvious at a glance that the container is healthy.
st.success(f"Streamlit container is running. Environment: **{app_env}**")

# st.info renders a blue info box.
st.info("Backend API and other services will be wired up in later phases.")