from datetime import datetime

import requests
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="IKMS Multi Agent RAG",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .history-badge {
        background-color: #f0f2f6;
        color: #31333F;
        padding: 4px 10px;
        border-radius: 16px;
        font-size: 12px;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        border: 1px solid #e0e2e6;
        margin-top: 5px;
        margin-bottom: 20px;
        width: fit-content;
    }

    .session-card {
        display: flex;
        align-items: center;
        justify-content: space-between;
        width: 100%;
        padding: 8px 12px;
        margin-bottom: 8px;
        background-color: var(--secondary-background-color);
        border: 1px solid transparent;
        border-radius: 8px;
        transition: all 0.2s ease;
        text-decoration: none !important;
        color: var(--text-color) !important;
    }

    .session-card:hover {
        border-color: #4A4E57;
    }

    .session-card.active {
        background-color: #4A4E57;
        border-color: #4A4E57;
    }

    .session-card.active .session-title {
        color: white !important;
    }

    .session-title {
        flex-grow: 1;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        font-size: 15px;
        font-weight: 500;
        color: var(--text-color);
        text-decoration: none;
        margin-right: 10px;
        cursor: pointer;
        color: white !important;
        text-decoration: none !important;
    }

    .session-delete {
        color: white !important;
        font-size: 20px;
        font-weight: bold;
        text-decoration: none;
        cursor: pointer;
        padding: 0 4px;
        line-height: 1;
        border-radius: 4px;
        text-decoration: none !important;

        opacity: 0;
        transition: opacity 0.2s ease, color 0.2s ease;
    }

    .session-card:hover .session-delete,
    .session-card.active .session-delete {
        opacity: 1;
        color: white !important;
    }

    .session-card .session-delete:hover {
        color: #ff4b4b !important;
        background-color: transparent !important;
    }

    [data-testid="stPopover"]:last-of-type {
        position: fixed !important;
        top: 65px !important;    
        right: 5.5rem !important;  
        z-index: 99999 !important;
        width: auto !important;
    }

    @media only screen and (max-width: 768px) {
        [data-testid="stPopover"]:last-of-type {
            right: 1rem !important; 
            top: 60px !important;
        }
    }

    [data-testid="stPopover"]:last-of-type button {
        background-color: white !important;
        color: #31333F !important;
        border: 1px solid #e0e2e6 !important;
        box-shadow: 0px 2px 6px rgba(0,0,0,0.1) !important;
        border-radius: 8px !important;
        padding-left: 15px !important;
        padding-right: 15px !important;
        height: 40px !important;
        font-weight: 500 !important;
        transition: all 0.2s ease;
    }

    [data-testid="stPopover"]:last-of-type button:hover {
        transform: translateY(-1px);
        background-color: #f0f2f6 !important;
        border-color: #d0d2d6 !important;
        color: #31333F !important; 
        box-shadow: 0px 4px 10px rgba(0,0,0,0.15) !important;
    }

    @media (prefers-color-scheme: dark) {
        [data-testid="stPopover"]:last-of-type button {
            background-color: #262730 !important;
            border-color: #4A4E57 !important;
            color: white !important;
            box-shadow: 0px 2px 6px rgba(0,0,0,0.4) !important;
        }

        [data-testid="stPopover"]:last-of-type button:hover {
            background-color: #31333F !important;
            border-color: #60646e !important;
            color: white !important;
        }
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 8rem;
    }
</style>
""", unsafe_allow_html=True)

if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {}
if "active_session_id" not in st.session_state:
    st.session_state.active_session_id = None
if "documents" not in st.session_state:
    st.session_state.documents = []
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
if "pending_delete_id" not in st.session_state:
    st.session_state.pending_delete_id = None


def fetch_documents():
    try:
        response = requests.get(f"{API_URL}/documents")
        if response.status_code == 200:
            st.session_state.documents = response.json().get("documents", [])
    except Exception:
        st.session_state.documents = []


def delete_document_api(filename):
    try:
        response = requests.delete(f"{API_URL}/documents/{filename}")
        if response.status_code == 200:
            if filename in st.session_state.documents:
                st.session_state.documents.remove(filename)

            st.toast(f"Deleted {filename}", icon="🗑️")
            st.rerun()
        else:
            st.error("Failed to delete document.")
    except Exception as e:
        st.error(f"Error: {e}")


def fetch_chat_sessions():
    try:
        response = requests.get(f"{API_URL}/sessions")
        if response.status_code == 200:
            sessions_list = response.json().get("sessions", [])
            current_ids = set()
            for s in sessions_list:
                s_id = s["id"]
                current_ids.add(s_id)
                if s_id not in st.session_state.chat_sessions:
                    st.session_state.chat_sessions[s_id] = {
                        "title": s["title"],
                        "last_updated": s["last_updated"],
                        "history": [],
                        "summary": None
                    }
                else:
                    st.session_state.chat_sessions[s_id]["title"] = s["title"]
                    st.session_state.chat_sessions[s_id]["last_updated"] = s["last_updated"]

            keys_to_remove = [k for k in st.session_state.chat_sessions if k not in current_ids]
            for k in keys_to_remove:
                del st.session_state.chat_sessions[k]
                if st.session_state.active_session_id == k:
                    st.session_state.active_session_id = None
    except Exception as e:
        pass


def fetch_session_history(session_id):
    try:
        response = requests.get(f"{API_URL}/sessions/{session_id}")
        if response.status_code == 200:
            data = response.json()
            if session_id in st.session_state.chat_sessions:
                st.session_state.chat_sessions[session_id]["history"] = data.get("history", [])
                st.session_state.chat_sessions[session_id]["summary"] = data.get("conversation_summary")
                return True
    except Exception:
        return False
    return False


def delete_chat_session_api(session_id):
    try:
        response = requests.delete(f"{API_URL}/sessions/{session_id}")
        if response.status_code == 200:
            st.session_state.pending_delete_id = None

            if session_id in st.session_state.chat_sessions:
                del st.session_state.chat_sessions[session_id]

            if st.session_state.active_session_id == session_id:
                st.session_state.active_session_id = None

            st.toast("Conversation deleted", icon="🗑️")
            st.rerun()
        else:
            st.error("Failed to delete conversation.")
    except Exception as e:
        st.error(f"Error: {e}")


def start_new_chat():
    st.session_state.active_session_id = None


def truncate_text(text, max_chars=22):
    if not text:
        return ""
    if len(text) > max_chars:
        return text[:max_chars] + "..."
    return text


def handle_query_params():
    params = st.query_params

    if "select_session" in params:
        session_id = params["select_session"]
        st.session_state.active_session_id = session_id

        st.query_params.clear()
        st.rerun()

    if "delete_prompt" in params:
        session_id = params["delete_prompt"]
        st.session_state.pending_delete_id = session_id

        st.query_params.clear()
        st.rerun()


@st.dialog("Confirm Deletion")
def confirm_delete_dialog(filename):
    st.write(f"Are you sure you want to delete **{filename}**?")
    st.warning("This action cannot be undone. The file will be completely removed.")
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Cancel", use_container_width=True):
            st.rerun()
    with col2:
        if st.button("Yes, Delete", type="primary", use_container_width=True):
            delete_document_api(filename)


@st.dialog("Delete Conversation")
def confirm_delete_chat_dialog():
    session_id = st.session_state.pending_delete_id
    title = "this conversation"
    if session_id in st.session_state.chat_sessions:
        title = st.session_state.chat_sessions[session_id].get("title", "this conversation")

    st.write(f"Are you sure you want to delete **{title}**?")
    st.warning("This conversation history will be permanently lost.")

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Cancel", use_container_width=True, key="cancel_chat_del"):
            st.session_state.pending_delete_id = None
            st.rerun()
    with col2:
        if st.button("Yes, Delete", type="primary", use_container_width=True, key="confirm_chat_del"):
            delete_chat_session_api(session_id)


handle_query_params()
fetch_documents()
fetch_chat_sessions()

if st.session_state.pending_delete_id:
    confirm_delete_chat_dialog()

with st.sidebar:
    st.title("IKMS Assistant")

    if st.button("Start a New Conversation", use_container_width=True, type="secondary"):
        start_new_chat()

    st.divider()

    st.subheader("Knowledge Base")
    with st.expander("Upload Documents", expanded=False):
        uploaded_file = st.file_uploader(
            "Upload PDF",
            type="pdf",
            label_visibility="collapsed",
            key=f"uploader_{st.session_state.uploader_key}"
        )

        if uploaded_file is not None:
            if st.button("Index Document", use_container_width=True):
                with st.status("Processing...", expanded=True) as status:
                    try:
                        files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
                        status.write("Uploading...")
                        response = requests.post(f"{API_URL}/index-pdf", files=files)
                        if response.status_code == 200:
                            status.update(label="Indexed!", state="complete", expanded=False)
                            st.toast(f"Indexed {uploaded_file.name}", icon="✅")
                            st.session_state.uploader_key += 1
                            fetch_documents()
                            st.rerun()
                        else:
                            status.update(label="Failed", state="error")
                            st.error(f"Error: {response.text}")
                    except Exception as e:
                        status.update(label="Error", state="error")
                        st.error(f"Connection Error: {e}")

    if st.session_state.documents:
        st.caption(f"{len(st.session_state.documents)} documents indexed")
        with st.expander("Manage Files", expanded=False):
            for doc in st.session_state.documents:
                col1, col2 = st.columns([0.85, 0.15], gap="small", vertical_alignment="center")
                with col1:
                    short_name = truncate_text(doc, max_chars=25)
                    st.markdown(f"<span title='{doc}' style='font-size: 0.9em;'>{short_name}</span>",
                                unsafe_allow_html=True)
                with col2:
                    if st.session_state.pending_delete_id is None:
                        if st.button("✕", key=f"del_{doc}", help=f"Delete {doc}", type="tertiary"):
                            confirm_delete_dialog(doc)
    else:
        st.caption("No documents found.")

    st.divider()

    st.subheader("Recent Conversations")

    sorted_sessions = sorted(
        st.session_state.chat_sessions.items(),
        key=lambda x: x[1].get('last_updated', ''),
        reverse=True
    )

    if not sorted_sessions:
        st.caption("No chat history.")
    else:
        html_content = ""
        for session_id, session_data in sorted_sessions:
            title = truncate_text(session_data.get("title", "New Chat"), max_chars=30)
            is_active = (session_id == st.session_state.active_session_id)
            active_class = "active" if is_active else ""

            row_html = f"""
            <div class="session-card {active_class}">
                <a href="?select_session={session_id}" target="_self" class="session-title" title="{session_data.get('title')}">
                    {title}
                </a>
                <a href="?delete_prompt={session_id}" target="_self" class="session-delete" title="Delete conversation">
                    &times;
                </a>
            </div>
            """
            html_content += row_html

        st.markdown(html_content, unsafe_allow_html=True)

if not st.session_state.documents:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(
            """
            <div style="text-align: center; margin-top: 50px;">
                <h1>Knowledge Base Empty</h1>
                <p style="font-size: 1.2em; color: #555;">
                    The system currently has no documents. 
                    <br>Please upload a PDF document in the sidebar to initialize the Knowledge Base.
                </p>
                <div style="padding: 20px; border-radius: 10px; border: 1px dashed #ccc; margin-top: 30px;">
                    <strong>Use the sidebar to upload a PDF to get started.</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    st.stop()

if st.session_state.active_session_id is None:
    st.header("New Conversation")
    st.markdown("Ask questions about your uploaded documents.")
    current_history = []
else:
    active_id = st.session_state.active_session_id
    if active_id in st.session_state.chat_sessions:
        if not st.session_state.chat_sessions[active_id].get("history"):
            with st.spinner("Loading conversation history..."):
                fetch_session_history(active_id)

        session_data = st.session_state.chat_sessions[active_id]
        st.header(session_data.get("title", "Chat"))
        current_history = session_data.get("history", [])
    else:
        st.session_state.active_session_id = None
        st.rerun()

for i, turn in enumerate(current_history):
    with st.chat_message("user"):
        st.markdown(turn["question"])

    with st.chat_message("assistant"):
        st.markdown(turn["answer"])

        if turn.get("used_history") or turn.get("context_used"):
            if turn.get("used_history"):
                st.markdown(
                    """
                    <div class="history-badge" title="Used context from previous turns">
                        <span>↺</span> Used History
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with st.expander("View Retrieval Context"):
                if turn.get("context_used"):
                    st.markdown("**Retrieved Document Chunks:**")
                    st.code(turn["context_used"], language="text")
                else:
                    st.markdown("*No document context was required for this answer.*")

                if turn.get("timestamp"):
                    try:
                        timestamp_obj = datetime.fromisoformat(turn.get('timestamp'))
                        st.caption(f"Generated at: {timestamp_obj.strftime('%B %d, %Y at %I:%M:%S %p')}")
                    except:
                        pass

if st.session_state.active_session_id:
    if st.session_state.active_session_id in st.session_state.chat_sessions:
        sess = st.session_state.chat_sessions[st.session_state.active_session_id]
        with st.popover("Chat Summary", use_container_width=False):
            if sess.get("summary"):
                st.markdown("### Conversation Summary")
                st.info(sess["summary"])
            else:
                st.markdown("### No Summary Yet")
                st.caption("A summary will be generated after 3 human messages.")

if prompt := st.chat_input("Ask a question..."):
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.status("Thinking...", expanded=True) as status:
            try:
                payload = {
                    "question": prompt,
                    "session_id": st.session_state.active_session_id
                }

                status.write("Generating Response...")
                response = requests.post(f"{API_URL}/qa/conversation", json=payload)
                response.raise_for_status()

                data = response.json()
                status.update(label="Response generated", state="complete", expanded=False)

                new_session_id = data["session_id"]
                st.session_state.active_session_id = new_session_id

                if new_session_id not in st.session_state.chat_sessions:
                    st.session_state.chat_sessions[new_session_id] = {
                        "history": [],
                        "title": data.get("session_title", "New Chat"),
                        "summary": None,
                        "last_updated": ""
                    }

                st.session_state.chat_sessions[new_session_id]["history"] = data["history"]
                st.session_state.chat_sessions[new_session_id]["summary"] = data.get("conversation_summary")
                st.session_state.chat_sessions[new_session_id]["title"] = data.get("session_title", "New Chat")

                if data["history"]:
                    st.session_state.chat_sessions[new_session_id]["last_updated"] = data["history"][-1].get(
                        "timestamp", "")

                st.rerun()

            except Exception as e:
                status.update(label="Error", state="error")
                st.error(f"Connection Error: {str(e)}")
