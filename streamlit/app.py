# app.py — Intelligent Document Processing Platform
# Phase 10: Streamlit Frontend
#
# Streamlit's execution model: this entire script runs top-to-bottom
# on every user interaction (button click, text input, file upload).
# Local variables are destroyed between runs. st.session_state is the
# only data store that survives across reruns for a given browser tab.

# ── Imports ────────────────────────────────────────────────────────────

import time                        # time.sleep() inside polling loop
import json                        # parse JSON bodies from API responses
import datetime                    # timestamp for uploaded_docs entries

import httpx                       # synchronous HTTP client for REST calls
import websockets.sync.client      # synchronous WebSocket client for chat streaming
import streamlit as st             # the entire UI framework


# ── Page configuration ─────────────────────────────────────────────────
# Must be the very first Streamlit call in the script.
# Any st.* call before this raises StreamlitAPIException.
# Calling it a second time in the same run also raises — so it must be
# at module level, not inside a function that could be called twice.

st.set_page_config(
    page_title="IDPP",             # text shown in the browser tab
    layout="wide",                 # use full browser width
    initial_sidebar_state="expanded",  # sidebar open on load
)


# ── Constants ──────────────────────────────────────────────────────────
# Hard-coded once here so every function below reads a name, not a string
# literal. If the nginx hostname ever changes (it won't in this project,
# but defensively) there is exactly one place to update.
#
# Why http://nginx/ and not http://localhost/?
# This script runs inside the Streamlit container on idp_network.
# "localhost" inside that container resolves to the Streamlit container
# itself — not the host machine, not nginx. Docker's embedded DNS
# resolves service names to container IPs on the shared bridge network.
# "nginx" resolves to the nginx container, which is what we want.

API_BASE = "http://nginx/api"      # all REST calls go through nginx
WS_BASE  = "ws://nginx/ws"        # all WebSocket connections go through nginx

# Status polling parameters — defined as constants so the numbers are
# visible and interview-explainable in one place.
POLL_INTERVAL_SECONDS = 2         # how long to wait between status checks
POLL_MAX_ATTEMPTS     = 30        # 30 × 2s = 60s maximum wait before timeout


# ── Session state initialisation ───────────────────────────────────────
# Called unconditionally as the first logic statement of every rerun.
# The "if key not in" guard is essential: without it this function would
# reset session_token to None on rerun #2, logging the user out the
# moment they click anything after connecting.
#
# Why a function rather than inline code?
# Clarity and testability. The initialisation block is named and
# self-contained. If we add a key later there is exactly one place to
# add it.

def init_session_state() -> None:
    defaults = {
        "session_token": None,   # str | None — None means not authenticated
        "tenant_name":   None,   # str | None — display only, never used for logic
        "uploaded_docs": [],     # list[dict] — one entry per upload this session
        "chat_history":  [],     # list[dict] — one entry per message this session
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()             # runs on every rerun, safe due to the guard


# ── Top-level UI branch ────────────────────────────────────────────────
# This single condition determines everything the user sees.
# We derive auth state from session_token alone — no separate flag —
# so the unauthenticated and invalid states are identical by construction.
#
# Alternative considered: st.experimental_dialog for login.
# Rejected: "no experimental features" constraint, and a sidebar login
# is simpler and more appropriate for a platform tool than a modal.

if st.session_state.session_token is None:

    # ── UNAUTHENTICATED STATE ──────────────────────────────────────
    # Show only what an unauthenticated user needs: a login form.
    # Nothing else renders. The sidebar and main panel are intentionally
    # blank except for the login block.

    with st.sidebar:
        st.header("🔐 Identity")
        st.caption("Enter your tenant passphrase to connect.")

        # Placeholder functions — replaced in Step 5.
        # We render the widgets now so the layout is verifiable in the
        # browser before any API logic exists.
        st.text_input("Passphrase", type="password", key="passphrase_input")
        st.button("Connect")

    # Main panel message — tells the user what to do.
    # col layout is established here even in the unauth state so the
    # page does not visually shift when login succeeds.
    st.info("👈 Enter your passphrase in the sidebar to get started.")

else:

    # ── AUTHENTICATED STATE ────────────────────────────────────────
    # Everything the authenticated user sees lives in this block.
    # Divided into: sidebar (identity + documents) and main panel
    # (document dashboard left, chat right).

    with st.sidebar:
        st.header("🔐 Identity")
        st.caption(f"Connected as: **{st.session_state.tenant_name}**")
        st.button("Disconnect")    # logic wired in Step 5

        st.divider()

        st.header("📄 Documents")
        # File uploader and document list — wired in Step 6.
        st.caption("Upload a document to begin.")

    # Main panel: two columns.
    # Ratio [1, 2] gives the document dashboard one third of the width
    # and the chat panel two thirds.
    # Alternative ratio considered: [1, 1] equal split.
    # Rejected: chat needs more horizontal space for readable message
    # bubbles; the document list is a narrow list, not a content area.
    col_docs, col_chat = st.columns([1, 2])

    with col_docs:
        st.subheader("📋 Document Dashboard")
        # Document list — wired in Step 6.
        st.caption("No documents uploaded yet.")

    with col_chat:
        st.subheader("💬 Chat")
        # Chat history and input — wired in Step 7.
        st.caption("Upload a document, then ask a question.")