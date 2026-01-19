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
        width: 110px
    }
    
    .history-badge svg {
        fill: #31333F;
        width: 12px;
        height: 12px;
    }

    .stButton button {
        text-align: left;
        padding-left: 15px;
    }
</style>
""", unsafe_allow_html=True)

if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {}

if "active_session_id" not in st.session_state:
    st.session_state.active_session_id = None


def start_new_chat():
    st.session_state.active_session_id = None


def select_chat(session_id):
    st.session_state.active_session_id = session_id


def truncate_title(title, max_chars=28):
    if not title:
        return "New Chat"
    if len(title) > max_chars:
        return title[:max_chars] + "..."
    return title


with st.sidebar:
    st.title("IKMS Multi Agent RAG")

    if st.button("New Conversation", use_container_width=True, type="primary"):
        start_new_chat()

    st.divider()

    st.subheader("Knowledge Base")
    with st.expander("Upload Documents", expanded=False):
        uploaded_file = st.file_uploader("Upload PDF", type="pdf", label_visibility="collapsed")

        if uploaded_file is not None:
            if st.button("Index Document", use_container_width=True):
                with st.status("Processing PDF...", expanded=True) as status:
                    try:
                        files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
                        status.write("Uploading to server...")

                        response = requests.post(f"{API_URL}/index-pdf", files=files)

                        if response.status_code == 200:
                            status.write("Indexing vectors...")
                            status.update(label="Upload Successful!", state="complete", expanded=False)
                            st.toast(f"Successfully indexed {uploaded_file.name}", icon="✅")
                        else:
                            status.update(label="Upload Failed", state="error")
                            st.error(f"Error: {response.text}")

                    except Exception as e:
                        status.update(label="Connection Error", state="error")
                        st.error(f"Failed to connect to backend: {str(e)}")

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
        button_type = "secondary"
        if session_id == st.session_state.active_session_id:
            button_type = "primary"

        display_title = truncate_title(session_data.get("title", "New Chat"))

        if st.button(display_title, key=session_id, use_container_width=True, type=button_type,
                     help=session_data.get("title")):
            select_chat(session_id)
            st.rerun()

    st.divider()

    if st.session_state.active_session_id:
        current_data = st.session_state.chat_sessions.get(st.session_state.active_session_id)
        if current_data:
            with st.expander("Chat Summary", expanded=False):
                if current_data.get("summary"):
                    st.info(current_data["summary"])
                    st.caption("Auto-generated summary of previous turns.")
                else:
                    st.caption("No summary generated yet, generated after 3 chats.")

if st.session_state.active_session_id is None:
    st.header("New Conversation")
    st.markdown("Upload a PDF in the sidebar or start asking questions below.")
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
            cols = st.columns([1, 10])

            with cols[0]:
                if turn.get("used_history"):
                    st.markdown(
                        """
                        <div class="history-badge" title="This answer used information from previous conversation turns">
                            <span>↺</span> Used History
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            with st.expander("View References"):
                if turn.get("context_used"):
                    st.markdown("**Retrieved Document Chunks:**")
                    st.code(turn["context_used"], language="text")
                else:
                    st.markdown("**Answer generated from chat history**")
                if turn.get("timestamp"):
                    timestamp_obj = datetime.fromisoformat(turn.get('timestamp'))
                    st.caption(f"Generated at: {timestamp_obj.strftime("%B %d, %Y at %I:%M:%S %p")}")

if prompt := st.chat_input("Ask a question about your documents..."):
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
