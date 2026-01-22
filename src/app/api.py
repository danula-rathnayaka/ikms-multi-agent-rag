import datetime
import os
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Any

from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse

from .core.agents.agents import generate_chat_title
from .core.agents.graph import run_conversational_qa_flow
from .core.retrieval.vector_store import index_documents, delete_document_vectors, delete_all_vectors
from .models import ConversationalQAResponse, ConversationalQARequest, ConversationHistory

UPLOAD_DIR = Path("/tmp/uploads")


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Server Starting... Cleaning up...")

    delete_all_vectors()

    if UPLOAD_DIR.exists():
        shutil.rmtree(UPLOAD_DIR)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    print("System Ready: Clean Slate.")

    yield

    print("Server Shutting Down...")


app = FastAPI(
    title="Class 12 Multi-Agent RAG Demo",
    description=(
        "Demo API for asking questions about a vector databases paper. "
        "The `/qa` endpoint currently returns placeholder responses and "
        "will be wired to a multi-agent RAG pipeline in later user stories."
    ),
    version="0.2.0",
    lifespan=lifespan
)

SESSIONS: Dict[str, Dict[str, Any]] = {}


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.exception_handler(Exception)
async def unhandled_exception_handler(
        request: Request, exc: Exception
) -> JSONResponse:  # pragma: no cover - simple demo handler
    """Catch-all handler for unexpected errors.

    FastAPI will still handle `HTTPException` instances and validation errors
    separately; this is only for truly unexpected failures so API consumers
    get a consistent 500 response body.
    """

    if isinstance(exc, HTTPException):
        # Let FastAPI handle HTTPException as usual.
        raise exc

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"},
    )


@app.get("/sessions", status_code=status.HTTP_200_OK)
async def list_sessions() -> dict:
    session_list = []
    for session_id, data in SESSIONS.items():
        last_updated = data.get("last_updated")
        if not last_updated and data.get("history"):
            last_updated = data["history"][-1].get("timestamp")

        session_list.append({
            "id": session_id,
            "title": data.get("title", "Untitled Session"),
            "last_updated": last_updated or datetime.datetime.min.isoformat()
        })

    session_list.sort(key=lambda x: x["last_updated"], reverse=True)
    return {"sessions": session_list}


@app.get("/sessions/{session_id}", status_code=status.HTTP_200_OK)
async def get_session(session_id: str) -> dict:
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found")

    session_data = SESSIONS[session_id]
    return {
        "id": session_id,
        "title": session_data.get("title"),
        "history": session_data.get("history", []),
        "conversation_summary": session_data.get("conversation_summary"),
        "last_updated": session_data.get("last_updated")
    }


@app.delete("/sessions/{session_id}", status_code=status.HTTP_200_OK)
async def delete_session(session_id: str) -> dict:
    if session_id in SESSIONS:
        del SESSIONS[session_id]
        return {"message": "Session deleted successfully"}
    raise HTTPException(status_code=404, detail="Session not found")


@app.post("/qa/conversation", response_model=ConversationalQAResponse)
async def conversational_qa(payload: ConversationalQARequest) -> ConversationalQAResponse:
    question = payload.question.strip()
    session_id = payload.session_id

    history_list = []
    if session_id and session_id in SESSIONS:
        history_list = SESSIONS[session_id]["history"]

    final_state = run_conversational_qa_flow(
        question=question,
        history=history_list,
        session_id=session_id
    )

    new_answer = final_state.get("answer", "")
    current_session_id = final_state.get("session_id")
    used_history = final_state.get("used_history", False)
    timestamp = datetime.datetime.now().isoformat()

    if current_session_id not in SESSIONS:
        SESSIONS[current_session_id] = {
            "title": "New Chat",
            "history": [],
            "conversation_summary": None,
            "last_updated": timestamp
        }

    if len(SESSIONS[current_session_id]["history"]) == 0:
        try:
            new_title = generate_chat_title(question, new_answer)
            SESSIONS[current_session_id]["title"] = new_title
        except Exception as e:
            print(f"Title generation failed: {e}")
            SESSIONS[current_session_id]["title"] = "New Conversation"

    new_turn = {
        "turn": len(SESSIONS[current_session_id]["history"]) + 1,
        "question": question,
        "answer": new_answer,
        "context_used": final_state.get("context", ""),
        "used_history": used_history,
        "timestamp": timestamp
    }

    SESSIONS[current_session_id]["history"].append(new_turn)
    SESSIONS[current_session_id]["last_updated"] = timestamp

    if final_state.get("conversation_summary"):
        SESSIONS[current_session_id]["conversation_summary"] = final_state.get("conversation_summary")

    return ConversationalQAResponse(
        answer=new_answer,
        session_id=current_session_id,
        session_title=SESSIONS[current_session_id]["title"],
        history=SESSIONS[current_session_id]["history"],
        conversation_summary=final_state.get("conversation_summary")
    )


@app.get("/qa/session/{session_id}/history", response_model=ConversationHistory)
async def get_conversation_history(session_id: str) -> ConversationHistory:
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found")

    return ConversationHistory(
        session_id=session_id,
        history=SESSIONS[session_id]["history"]
    )


@app.post("/index-pdf", status_code=status.HTTP_200_OK)
async def index_pdf(file: UploadFile = File(...)) -> dict:
    """Upload a PDF and index it into the vector database.

    This endpoint:
    - Accepts a PDF file upload
    - Saves it to the local `data/uploads/` directory
    - Uses PyPDFLoader to load the document into LangChain `Document` objects
    - Indexes those documents into the configured Pinecone vector store
    """
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported.",
        )

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    file_path = UPLOAD_DIR / file.filename

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    chunks_indexed = index_documents(file_path)

    return {
        "filename": file.filename,
        "chunks_indexed": chunks_indexed,
        "message": "PDF indexed successfully.",
    }


@app.get("/documents", status_code=status.HTTP_200_OK)
async def list_documents() -> dict:
    files = []
    if UPLOAD_DIR.exists():
        files = [f.name for f in UPLOAD_DIR.glob("*.pdf")]
    return {"documents": files}


@app.delete("/documents/{filename}", status_code=status.HTTP_200_OK)
async def delete_document(filename: str) -> dict:
    file_path = UPLOAD_DIR / filename

    success = delete_document_vectors(file_path)

    if file_path.exists():
        try:
            os.remove(file_path)
        except OSError as e:
            print(f"Error deleting local file: {e}")

    if not success and not file_path.exists():
        raise HTTPException(status_code=404, detail="Document not found.")

    return {"message": f"Document {filename} deleted."}
