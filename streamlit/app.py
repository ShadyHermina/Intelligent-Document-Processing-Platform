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
        "ws_connection": None,   # websockets connection | None — persistent across reruns
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()             # runs on every rerun, safe due to the guard

# ── Auth callbacks ─────────────────────────────────────────────────────
# These functions are passed to st.button(on_click=...).
# Streamlit invokes them BEFORE the script body runs on the rerun
# caused by their button click. This means by the time the
# if/else branch below executes, session_state is already updated.


def connect() -> None:
    # Read the access_phrase from session_state directly.
    # When a widget is given a key= argument, Streamlit mirrors its
    # current value into st.session_state[key] automatically.
    # We read it here rather than passing it as a parameter because
    # on_click callbacks receive no arguments from the button itself.
    access_phrase = st.session_state.get("access_phrase_input", "").strip()

    # Guard: empty access_phrase.
    # We check before making any network call. st.error() inside a
    # callback still renders — Streamlit queues the message and
    # displays it on the rerun that follows the callback.
    if not access_phrase:
        st.error("Please enter an access phrase.")
        return

    # Every network call is wrapped in try/except.
    # The constraint: the UI must never crash on API errors.
    # httpx.RequestError covers connection failures (nginx down,
    # DNS failure, timeout). httpx.HTTPStatusError is raised by
    # response.raise_for_status() when the server returns 4xx or 5xx.
    try:
        response = httpx.post(
            f"{API_BASE}/session/init",
            # The FastAPI endpoint expects a JSON body with one key.
            # httpx.post(json=...) serialises the dict and sets
            # Content-Type: application/json automatically.
            json={"access_phrase": access_phrase},
            # Timeout: how long to wait for the server to respond.
            # 10 seconds is generous for a local Docker network call
            # that should complete in milliseconds. Without a timeout,
            # a hung nginx would freeze this rerun indefinitely.
            timeout=10.0,
        )
        # raise_for_status() raises httpx.HTTPStatusError if the
        # response status is 4xx or 5xx. A 401 from FastAPI (wrong
        # access_phrase) is caught here and handled below, not crashed.
        response.raise_for_status()

        # .json() parses the response body as JSON and returns a dict.
        # We expect: {"session_token": "..."}
        data  = response.json()
        session_token = data["session_token"]

        # Second call: get tenant name using the token we just received.
        # GET /session/me requires the Authorization header — this is
        # the first use of the Bearer token pattern that every subsequent
        # authenticated call will also follow.
        me_response = httpx.get(
            f"{API_BASE}/session/me",
            headers={"Authorization": f"Bearer {session_token}"},
            timeout=10.0,
        )
        me_response.raise_for_status()
        me_data = me_response.json()

        # Write both values to session_state only after both calls
        # succeed. If the second call fails, session_token stays None
        # and the user stays on the login screen — no half-authenticated
        # state is possible.
        st.session_state.session_token = session_token
        st.session_state.tenant_name   = me_data["tenant_name"]

    except httpx.HTTPStatusError as e:
        # The server responded but with an error status.
        # 401 means wrong access_phrase. We show a clear message and
        # do not write anything to session_state — the user stays
        # on the login screen.
        if e.response.status_code == 401:
            st.error("Incorrect access phrase. Please try again.")
        else:
            # Any other HTTP error (500, 503, etc.) — show the code
            # so the user knows it's a server problem, not their input.
            st.error(f"Server error: {e.response.status_code}. Please try again.")

    except httpx.RequestError:
        # Network-level failure: nginx unreachable, DNS failure, timeout.
        # We do not expose the raw exception message to the user —
        # it contains internal hostnames (nginx) that are meaningless
        # to a tenant and would look like a bug.
        st.error("Could not reach the server. Please try again.")


def disconnect() -> None:
    # Full wipe of all five keys — not just the token.
    # Rationale locked in Step 2: if a different tenant logs in on
    # the same tab after disconnect, stale documents and chat history
    # from the previous tenant must not be visible.
    # We reset to the same defaults as init_session_state() rather
    # than calling del, because del would cause init_session_state()
    # to recreate them as empty defaults on the next rerun anyway —
    # same result, one fewer step.
    #
    # WebSocket is closed BEFORE wiping state — if we set ws_connection
    # to None without closing first, the server-side connection stays
    # open until it times out: a resource leak on the FastAPI side.
    _close_ws()
    st.session_state.session_token = None
    st.session_state.tenant_name   = None
    st.session_state.uploaded_docs = []
    st.session_state.chat_history  = []

# ── WebSocket helpers ──────────────────────────────────────────────────

def _get_or_open_ws():
    """
    Return the existing WebSocket connection if healthy, or open a new one.

    Why persistent connection rather than per-message?
    The FastAPI chat endpoint maintains conversation history in memory
    for the duration of the connection (chat.py line 693: history = []).
    A new connection starts with empty history — the server forgets all
    prior turns. To get conversation memory across turns, we must reuse
    the same connection.

    Why store in session_state?
    Streamlit reruns the script on every user interaction, destroying all
    local variables. Storing the connection in session_state is the only
    way to keep it alive across the rerun caused by the user pressing Send.

    Liveness check: we attempt a ping before reusing the connection.
    If the connection is stale (server restarted, timeout), the ping
    raises and we open a fresh one — the user loses that session's
    history but does not see a crash.
    """
    ws = st.session_state.get("ws_connection")

    if ws is not None:
        # Verify the connection is still alive with a ping.
        # websockets.sync.client connections expose .ping() which sends
        # a WebSocket ping frame and waits for a pong. If the connection
        # is dead, this raises ConnectionClosed or OSError.
        try:
            ws.ping()
            return ws   # healthy — reuse it
        except Exception:
            # Connection is stale — fall through to open a new one.
            st.session_state.ws_connection = None

    # Open a new WebSocket connection.
    # Token goes in the query param — the WebSocket endpoint reads it
    # from websocket.query_params, not from an Authorization header,
    # because the WebSocket handshake is an HTTP GET with no body.
    token = st.session_state.session_token
    url   = f"{WS_BASE}/chat?token={token}"

    try:
        ws = websockets.sync.client.connect(
            url,
            # open_timeout: how long to wait for the server to accept
            # the WebSocket handshake. 10 seconds is generous for a
            # local Docker network call.
            open_timeout=10,
        )
        st.session_state.ws_connection = ws
        return ws
    except Exception as e:
        raise ConnectionError(f"Could not connect to chat server: {e}")


def _close_ws() -> None:
    """
    Close the WebSocket connection if one exists and clear it from state.
    Called on disconnect and on unrecoverable stream errors.
    Safe to call when ws_connection is already None.
    """
    ws = st.session_state.get("ws_connection")
    if ws is not None:
        try:
            ws.close()
        except Exception:
            # Already closed or broken — nothing to do.
            pass
        st.session_state.ws_connection = None


def send_chat_message(user_message: str) -> None:
    """
    Send a message over the persistent WebSocket and stream the response
    into the chat panel token-by-token.

    Flow:
      1. Append user message to chat_history immediately — renders above
         the streaming response so the user sees their own message while
         waiting for the backend.
      2. Get or open the WebSocket connection.
      3. Send the user message text.
      4. Receive text frames in a loop, accumulate into a string.
         Update an st.empty() placeholder on each frame.
      5. On [DONE] — finalize: append assistant message to chat_history,
         clear the placeholder, call st.rerun() to re-render from history.
      6. On any error — show st.error(), close the connection so the
         next send attempt opens a fresh one.

    Why append user message BEFORE the network call?
    If the backend takes 10 seconds, the user stares at a blank panel.
    Appending immediately gives instant feedback that the UI received
    the message, even while the backend processes it.

    Why clear the placeholder after [DONE]?
    The streaming placeholder and the chat_history render are two
    separate regions. Once the message is committed to chat_history,
    st.rerun() will render it from history. If we leave the placeholder
    visible, the message appears twice. Clearing it first avoids the
    duplicate.

    Why call st.rerun() at the end?
    The streaming loop ran inside one rerun. After it finishes, the
    script has reached the end of col_chat — new history entries won't
    render until the next rerun. st.rerun() triggers that immediately,
    so the finalized message appears without the user having to click
    anything.
    """
    # Step 1: append user message to history immediately.
    st.session_state.chat_history.append({
        "role":      "user",
        "content":   user_message,
        "citations": [],
    })

    # Step 2: get or open connection.
    try:
        ws = _get_or_open_ws()
    except ConnectionError as e:
        st.error(str(e))
        # Remove the user message we just appended — the turn failed
        # before the server saw it, so history would be inconsistent.
        st.session_state.chat_history.pop()
        return

    # Step 3: send the user message.
    try:
        ws.send(user_message)
    except Exception as e:
        st.error(f"Failed to send message: {e}")
        st.session_state.chat_history.pop()
        _close_ws()
        return

    # Step 4 + 5: stream response using st.write_stream().
    #
    # Why st.write_stream() instead of st.empty().markdown() in a loop?
    # Streamlit batches all st.* calls during a rerun and flushes them
    # to the browser only when the script yields control — which in a
    # tight recv() loop means only at the very end, making the response
    # appear all at once. st.write_stream() bypasses this batching by
    # pushing each chunk through Streamlit's internal delta queue
    # immediately as it is yielded, giving true word-by-word visibility.
    #
    # Why a generator function rather than a generator expression?
    # We need to intercept the [DONE] sentinel and handle connection
    # errors inside the loop. A generator function (yield inside def)
    # allows that cleanly. A generator expression does not.
    #
    # Why accumulate inside the generator via nonlocal?
    # st.write_stream() returns the full concatenated string when the
    # generator is exhausted — we could use that return value directly.
    # But we also need the accumulated text to detect __ERROR__ markers
    # set when an exception occurs inside the generator, where st.*
    # calls are not allowed. nonlocal gives us a shared reference that
    # both the generator and the outer function can read.

    accumulated = ""

    def token_generator():
        nonlocal accumulated
        try:
            while True:
                frame = ws.recv()
                # ws.recv() blocks until a text frame arrives.
                # Raises ConnectionClosed if the server closes.

                if frame == "\n\n[DONE]":
                    # End-of-response signal — stop the generator.
                    # StopIteration tells st.write_stream() we are done.
                    return

                accumulated += frame
                yield frame
                # yield pushes this chunk to st.write_stream() which
                # forwards it to the browser immediately — no batching.

        except Exception as e:
            # Cannot call st.error() inside a generator — Streamlit does
            # not allow st.* calls from generator context. We encode the
            # error in accumulated so the outer function can detect and
            # surface it after write_stream() returns.
            accumulated = f"__ERROR__:{e}"

    # st.write_stream() renders yielded chunks inside a chat_message
    # bubble in real time. It returns the full concatenated text when
    # the generator is exhausted.
    try:
        with st.chat_message("assistant"):
            st.write_stream(token_generator())
    except Exception as e:
        st.error(f"Connection lost while receiving response: {e}")
        _close_ws()
        st.session_state.chat_history.pop()
        return

    # Detect errors encoded by the generator.
    if accumulated.startswith("__ERROR__:"):
        st.error(accumulated.replace("__ERROR__:", "Connection lost: "))
        _close_ws()
        st.session_state.chat_history.pop()
        return

    # Step 5: commit completed message to chat_history.
    # accumulated holds the full response text without the [DONE] sentinel.
    # Citations are embedded inline by GPT-4o as [Source N: section='...',
    # page/location=N] markers within the text — no separate field needed.
    st.session_state.chat_history.append({
        "role":      "assistant",
        "content":   accumulated,
        "citations": [],
    })

    # Trigger a rerun so the history loop above re-renders all messages
    # including the one just committed. The st.write_stream() bubble
    # disappears on rerun but is immediately replaced by the history
    # render — the user sees no visual gap.
    st.rerun()

# ── Document helpers ───────────────────────────────────────────────────

def _fetch_doc_status(token: str, document_id: str) -> dict | None:
    """
    Call GET /api/documents/{id}/status and return the parsed dict.
    Returns None on any error — callers handle the None case.

    Used in two places:
      1. After a successful upload — to get filename and uploaded_at
         (the upload response doesn't include them).
      2. After a 409 duplicate — to fetch the full record of the
         previously uploaded file so we can display it correctly.

    Why a standalone helper rather than inline code?
    Both callers need identical error handling and identical field
    extraction. One function, two call sites, no duplication.
    """
    try:
        r = httpx.get(
            f"{API_BASE}/documents/{document_id}/status",
            headers={"Authorization": f"Bearer {token}"},
            timeout=10.0,
        )
        r.raise_for_status()
        return r.json()
    except (httpx.HTTPStatusError, httpx.RequestError):
        return None


def _status_badge(status: str) -> str:
    """
    Convert a pipeline status string to a display string with an emoji.

    Why emojis rather than colored boxes?
    st.markdown colored badges require unsafe_allow_html=True, which
    opens an XSS vector if any user-controlled text ever reaches a
    markdown call. Emojis are plain unicode — no HTML, no risk.
    """
    badges = {
        "pending":    "⏳ pending",
        "ingesting":  "⟳ ingesting",
        "classified": "🔍 classified",
        "embedded":   "✅ embedded",
        "failed":     "❌ failed",
    }
    # .get() with a fallback so an unexpected status value never crashes
    # the render loop — it just shows as-is with a neutral marker.
    return badges.get(status, f"• {status}")


def upload_document(file) -> None:
    """
    Upload a file to POST /api/documents/upload, handle all response
    cases, and append the result to st.session_state.uploaded_docs.

    Called from the Upload button's on_click callback. Receives the
    UploadedFile object from st.file_uploader.

    Why not inline in the button callback?
    The upload logic is ~60 lines. Keeping it in a named function makes
    the button callback a one-liner and keeps the UI layout section
    readable.

    Response cases handled:
      200 → success: append doc to uploaded_docs, show success message
      409 → duplicate: fetch existing record, append it, show info message
      400 → bad file: show error (should not reach here — we validate
            client-side first, but the server is the authoritative check)
      422 → unparseable file: show error with server's detail message
      any other → generic error
    """
    token = st.session_state.session_token

    # Read file bytes from the Streamlit UploadedFile object.
    # .read() returns bytes. .seek(0) resets the read pointer — not
    # needed here since we read once, but defensive for future changes.
    file_bytes = file.read()

    try:
        response = httpx.post(
            f"{API_BASE}/documents/upload",
            headers={"Authorization": f"Bearer {token}"},
            # httpx files= parameter sends a multipart/form-data request.
            # Tuple format: (filename, bytes, content_type).
            # The field name "file" must match the FastAPI parameter name
            # `file: UploadFile = File(...)` in the router.
            files={"file": (file.name, file_bytes, file.type)},
            # Long timeout: the pipeline runs synchronously server-side.
            # A real document can take 10-30 seconds to chunk, embed,
            # and classify. 120 seconds is generous but not infinite.
            timeout=120.0,
        )
        response.raise_for_status()

        # Success path — 200 response
        data = response.json()
        # Upload response: document_id, status, doc_type,
        #                  confidence, chunks_embedded
        # Missing from upload response: filename, uploaded_at
        # We get those from one status poll immediately after.
        doc_id = data["document_id"]

        status_data = _fetch_doc_status(token, doc_id)

        # Build the doc dict from both responses combined.
        # If the status poll fails (status_data is None), we fall back
        # to what the upload response gave us — partial data is better
        # than no data.
        doc_entry = {
            "document_id":     doc_id,
            "filename":        status_data["filename"] if status_data else file.name,
            "status":          data["status"],
            "doc_type":        data["doc_type"],
            "chunks_embedded": data["chunks_embedded"],
            "confidence":      data.get("confidence"),
            "uploaded_at":     (
                status_data["uploaded_at"] if status_data
                else datetime.datetime.now().isoformat()
            ),
        }
        st.session_state.uploaded_docs.append(doc_entry)
        st.success(
            f"✅ **{doc_entry['filename']}** uploaded successfully — "
            f"{data['chunks_embedded']} chunks embedded, "
            f"classified as **{data['doc_type']}** "
            f"(confidence: {data['confidence']:.0%})"
        )

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 409:
            # Duplicate file — the server returns the existing document_id.
            # We fetch its status and add it to the list so the user can
            # still use it for chat, even though it was previously uploaded.
            detail = e.response.json().get("detail", {})
            doc_id = detail.get("document_id")

            if doc_id:
                status_data = _fetch_doc_status(token, doc_id)
                if status_data:
                    # Check whether this doc is already in our session list
                    # to avoid adding the same document twice if the user
                    # uploads the same file again within one session.
                    existing_ids = [
                        d["document_id"]
                        for d in st.session_state.uploaded_docs
                    ]
                    if doc_id not in existing_ids:
                        doc_entry = {
                            "document_id":     doc_id,
                            "filename":        status_data["filename"],
                            "status":          status_data["status"],
                            "doc_type":        status_data.get("doc_type", "unknown"),
                            "chunks_embedded": None,   # not in status response
                            "uploaded_at":     status_data["uploaded_at"],
                        }
                        st.session_state.uploaded_docs.append(doc_entry)
                        st.info(
                            f"ℹ️ **{status_data['filename']}** was previously uploaded. "
                            f"Added to your document list."
                        )
                    else:
                        st.info(
                            f"ℹ️ **{status_data['filename']}** is already in your list."
                        )
            else:
                st.error("This file has already been uploaded.")

        elif e.response.status_code == 400:
            # Should not reach here — we validate the extension client-side.
            # But the server is authoritative, so we surface its message.
            st.error(f"Upload rejected: {e.response.json().get('detail', 'Bad request.')}")

        elif e.response.status_code == 422:
            # File was accepted but pipeline could not process it —
            # empty content, image-only PDF, corrupted file.
            st.error(
                f"Could not process **{file.name}**: "
                f"{e.response.json().get('detail', 'File could not be parsed.')}"
            )
        else:
            st.error(f"Upload failed with server error {e.response.status_code}.")

    except httpx.RequestError:
        st.error("Could not reach the server during upload. Please try again.")

def handle_upload() -> None:
    """
    on_click callback for the Upload button.

    Reads the uploaded file from session_state, validates the extension
    client-side, then calls upload_document().

    Why validate client-side if the server also validates?
    The server is the authoritative check — we cannot bypass it.
    The client-side check gives instant feedback without a network round
    trip. Both checks must exist: client for UX, server for security.

    Why a separate function from upload_document()?
    on_click callbacks receive no arguments. We need to read the file
    from session_state (where st.file_uploader writes it via key=),
    validate it, and only then call the actual upload logic. Separating
    the two keeps each function focused on one responsibility.
    """
    uploaded_file = st.session_state.get("file_uploader")

    if uploaded_file is None:
        st.error("Please select a file before clicking Upload.")
        return

    # Client-side extension check.
    allowed = {"pdf", "docx", "xlsx"}
    extension = uploaded_file.name.rsplit(".", 1)[-1].lower() if "." in uploaded_file.name else ""

    if extension not in allowed:
        st.error(
            f"**{uploaded_file.name}** is not supported. "
            f"Please upload a PDF, DOCX, or XLSX file."
        )
        return

    # Client-side size check: 20MB = 20 * 1024 * 1024 bytes.
    # Nginx enforces this at the network level (returns 413), but we
    # check here to give a clear message before the request is even sent.
    max_bytes = 20 * 1024 * 1024
    if uploaded_file.size > max_bytes:
        st.error(
            f"**{uploaded_file.name}** is too large "
            f"({uploaded_file.size / 1024 / 1024:.1f} MB). "
            f"Maximum size is 20 MB."
        )
        return

    with st.spinner(f"Processing {uploaded_file.name}…"):
        upload_document(uploaded_file)

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
        st.caption("Enter your tenant access phrase to connect.")

        # Placeholder functions — replaced in Step 5.
        # We render the widgets now so the layout is verifiable in the
        # browser before any API logic exists.
        st.text_input("Acess Phrase", type="password", key="access_phrase_input")
        
        st.button(
            "Connect",
            # on_click: Streamlit calls connect() before re-running
            # the script. By the time the if/else branch below is
            # evaluated, session_state already reflects the outcome.
            on_click=connect,
            # use_container_width makes the button fill the sidebar
            # width rather than sizing to its label text. Looks cleaner
            # and is easier to click on a narrow sidebar.
            use_container_width=True,
        )

    # Main panel message — tells the user what to do.
    # col layout is established here even in the unauth state so the
    # page does not visually shift when login succeeds.
    st.info("👈 Enter your access phrase in the sidebar to get started.")

else:

    # ── AUTHENTICATED STATE ────────────────────────────────────────
    # Everything the authenticated user sees lives in this block.
    # Divided into: sidebar (identity + documents) and main panel
    # (document dashboard left, chat right).

    with st.sidebar:
        st.header("🔐 Identity")
        st.caption(f"Connected as: **{st.session_state.tenant_name}**")

        st.button(
            "Disconnect",
            on_click=disconnect,
            use_container_width=True,
        )

        st.divider()

        st.header("📄 Documents")

        # File uploader widget.
        # type= restricts the file picker dialog on the client side.
        # This is a UX hint only — the server enforces the real limit.
        # key= mirrors the uploaded file into st.session_state["file_uploader"]
        # so handle_upload() can read it without needing a parameter.
        st.file_uploader(
            "Choose a file",
            type=["pdf", "docx", "xlsx"],
            key="file_uploader",
            label_visibility="collapsed",
        )

        st.button(
            "Upload",
            on_click=handle_upload,
            use_container_width=True,
        )

    # Main panel: two columns.
    # Ratio [1, 2] gives the document dashboard one third of the width
    # and the chat panel two thirds.
    # Alternative ratio considered: [1, 1] equal split.
    # Rejected: chat needs more horizontal space for readable message
    # bubbles; the document list is a narrow list, not a content area.
    col_docs, col_chat = st.columns([1, 2])

    with col_docs:
        st.subheader("📋 Document Dashboard")

        # Document list
        if not st.session_state.uploaded_docs:
            st.caption("No documents uploaded yet.")
        else:
            # Render each uploaded document as a card.
            # Newest first: reversed() iterates without making a copy.
            for doc in reversed(st.session_state.uploaded_docs):

                # st.container() groups the widgets for one document
                # visually. border=True draws a subtle box around it.
                with st.container(border=True):
                    # Filename — bold, no UUID ever shown.
                    st.markdown(f"**{doc['filename']}**")

                    # Status badge and doc_type on one line.
                    # doc_type may be None if status polling failed —
                    # we display "—" rather than the word "None".
                    doc_type_display = doc["doc_type"] or "—"
                    st.caption(
                        f"{_status_badge(doc['status'])}  ·  {doc_type_display}"
                    )

                    # Upload time — reformat ISO timestamp to HH:MM:SS.
                    # The full ISO string (2026-06-30T20:49:03.504729+00:00)
                    # is too long for a sidebar card.
                    try:
                        dt = datetime.datetime.fromisoformat(doc["uploaded_at"])
                        time_display = dt.strftime("%H:%M:%S UTC")
                    except (ValueError, TypeError):
                        time_display = "—"
                    st.caption(f"Uploaded: {time_display}")

                    # Expandable details panel.
                    # st.expander renders a collapsible section — collapsed
                    # by default so the card stays compact until the user
                    # wants more information.
                    # expanded=False is the default but stated explicitly
                    # for clarity — every card starts collapsed.
                    with st.expander("Details", expanded=False):

                        # Doc type — always available.
                        st.markdown(f"**Document type:** {doc['doc_type'] or '—'}")

                        # Confidence — only available for docs uploaded in
                        # this session. The upload response includes it;
                        # the status endpoint does not. Duplicate-path docs
                        # (added via 409) show — instead of crashing.
                        confidence = doc.get("confidence")
                        if confidence is not None:
                            st.markdown(f"**Confidence:** {confidence:.0%}")
                        else:
                            st.markdown("**Confidence:** —")

                        # Chunks embedded — same availability as confidence.
                        chunks = doc.get("chunks_embedded")
                        if chunks is not None:
                            st.markdown(f"**Chunks embedded:** {chunks}")
                        else:
                            st.markdown("**Chunks embedded:** —")

                        # Full upload timestamp — the card shows HH:MM:SS
                        # only; the expander shows the complete ISO string
                        # for precision.
                        st.markdown(f"**Uploaded:** {time_display}")

    with col_chat:
        st.subheader("💬 Chat")

        # ── Render chat history ────────────────────────────────────
        # Every completed message in chat_history is rendered here on
        # every rerun. Because history is in session_state it survives
        # reruns — this loop always produces the full conversation.
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                # st.chat_message("user") right-aligns the bubble and
                # shows a user avatar. Standard Streamlit chat UI pattern.
                with st.chat_message("user"):
                    st.markdown(msg["content"])
            else:
                # Assistant messages left-aligned with assistant avatar.
                with st.chat_message("assistant"):
                    st.markdown(msg["content"])
                    # Citations are embedded inline in msg["content"] as
                    # [Source N: section='...', page/location=N] markers
                    # written by GPT-4o. They render naturally within the
                    # markdown above — no separate rendering needed.

        # ── Chat input ─────────────────────────────────────────────
        # st.chat_input renders a fixed input bar at the bottom of the
        # column. Returns the submitted text on the rerun caused by
        # pressing Enter; returns None on all other reruns.
        #
        # Why st.chat_input instead of st.text_input + st.button?
        # st.chat_input is purpose-built for conversational UI: stays
        # fixed at the bottom, clears itself after submit, submits on
        # Enter. st.text_input needs a separate button and manual clearing.
        #
        # Why not use on_click here?
        # st.chat_input has no on_click callback — it returns the
        # submitted value directly. We check the return value inline.
        user_input = st.chat_input("Type your message and press Enter…")

        if user_input:
            # user_input is non-None only on the rerun caused by the
            # user pressing Enter. Calling send_chat_message() here means
            # the st.write_stream() generator runs inside this same rerun,
            # pushing each token to the browser immediately as it arrives.
            send_chat_message(user_input)