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

    [data-testid="stSidebar"] .stButton button {
        width: 100%;
        display: flex !important;
        justify-content: flex-start !important; /* Align container left */
        text-align: left !important;
        padding-left: 15px !important;
    }

    [data-testid="stSidebar"] .stButton button > div {
        display: flex !important;
        justify-content: flex-start !important;
        width: 100% !important;
    }

    [data-testid="stSidebar"] .stButton button p {
        text-align: left !important;
        width: 100%;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        display: block;
    }

    [data-testid="stSidebar"] button[kind="primary"] {
        background-color: #4A4E57 !important;
        border-color: #4A4E57 !important;
        color: white !important;
        font-weight: 600;
    }

    [data-testid="stSidebar"] button[kind="primary"]:hover {
        background-color: #31333F !important;
        border-color: #31333F !important;
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

    [data-testid="stPopover"]:last-of-type button:active, 
    [data-testid="stPopover"]:last-of-type button:focus {
        border-color: #4A4E57 !important;
        color: #4A4E57 !important;
        background-color: #e6e8eb !important;
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
            color: white !important; /* Force text white on hover */
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
            st.toast(f"Deleted {filename}", icon="🗑️")
            fetch_documents()
            st.rerun()
        else:
            st.error("Failed to delete document.")
    except Exception as e:
        st.error(f"Error: {e}")


def start_new_chat():
    st.session_state.active_session_id = None


def select_chat(session_id):
    st.session_state.active_session_id = session_id


def truncate_text(text, max_chars=22):
    if not text:
        return ""
    if len(text) > max_chars:
        return text[:max_chars] + "..."
    return text


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
            st.rerun()


fetch_documents()

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
                    short_name = truncate_text(doc, max_chars=20)
                    st.markdown(f"<span title='{doc}' style='font-size: 0.9em;'>{short_name}</span>",
                                unsafe_allow_html=True)
                with col2:
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

    for session_id, session_data in sorted_sessions:
        is_active = (session_id == st.session_state.active_session_id)
        btn_type = "primary" if is_active else "secondary"
        display_title = truncate_text(session_data.get("title", "New Chat"), max_chars=40)

        if st.button(display_title, key=session_id, use_container_width=True, type=btn_type,
                     help=session_data.get("title")):
            select_chat(session_id)
            st.rerun()

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
    session_data = st.session_state.chat_sessions[st.session_state.active_session_id]
    st.header(session_data.get("title", "Chat"))
    current_history = session_data.get("history", [])

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
    with st.popover("Chat Summary", use_container_width=False):
        if session_data.get("summary"):
            st.markdown("### Conversation Summary")
            st.info(session_data["summary"])
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
