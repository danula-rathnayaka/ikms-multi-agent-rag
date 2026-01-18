from pathlib import Path
from typing import Dict, List

from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse

from .core.agents.graph import run_conversational_qa_flow
from .models import ConversationalQAResponse, ConversationalQARequest, ConversationHistory
from .services.indexing_service import index_pdf_file

app = FastAPI(
    title="Class 12 Multi-Agent RAG Demo",
    description=(
        "Demo API for asking questions about a vector databases paper. "
        "The `/qa` endpoint currently returns placeholder responses and "
        "will be wired to a multi-agent RAG pipeline in later user stories."
    ),
    version="0.2.0",
)

SESSIONS: Dict[str, List[dict]] = {}


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


@app.post("/qa/conversation", response_model=ConversationalQAResponse)
async def conversational_qa(payload: ConversationalQARequest) -> ConversationalQAResponse:
    question = payload.question.strip()
    session_id = payload.session_id

    history = []
    if session_id:
        if session_id in SESSIONS:
            history = SESSIONS[session_id]
        else:
            history = []

    final_state = run_conversational_qa_flow(
        question=question,
        history=history,
        session_id=session_id
    )

    new_answer = final_state.get("answer", "")
    current_session_id = final_state.get("session_id")

    new_turn = {
        "question": question,
        "answer": new_answer,
        "context_used": final_state.get("context", "")
    }

    if current_session_id not in SESSIONS:
        SESSIONS[current_session_id] = []

    SESSIONS[current_session_id].append(new_turn)

    return ConversationalQAResponse(
        answer=new_answer,
        session_id=current_session_id,
        history=SESSIONS[current_session_id]
    )


@app.get("/qa/session/{session_id}/history", response_model=ConversationHistory)
async def get_conversation_history(session_id: str) -> ConversationHistory:
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found")

    return ConversationHistory(
        session_id=session_id,
        history=SESSIONS[session_id]
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

    if file.content_type not in ("application/pdf",):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported.",
        )

    upload_dir = Path("data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)

    file_path = upload_dir / file.filename
    contents = await file.read()
    file_path.write_bytes(contents)

    # Index the saved PDF
    chunks_indexed = index_pdf_file(file_path)

    return {
        "filename": file.filename,
        "chunks_indexed": chunks_indexed,
        "message": "PDF indexed successfully.",
    }
