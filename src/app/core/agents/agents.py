"""Agent implementations for the multi-agent RAG flow."""

from typing import List

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate  # <--- NEW IMPORT
from pydantic import BaseModel, Field  # <--- NEW IMPORTS

from .prompts import (
    RETRIEVAL_SYSTEM_PROMPT,
    SUMMARIZATION_SYSTEM_PROMPT,
    VERIFICATION_SYSTEM_PROMPT,
    MEMORY_SUMMARIZATION_SYSTEM_PROMPT,
    TITLE_GENERATION_PROMPT,
)
from .state import QAState
from .tools import retrieval_tool
from ..llm.factory import create_chat_model


def _extract_last_ai_content(messages: List[object]) -> str:
    """Extract the content of the last AIMessage in a messages list."""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            return str(msg.content)
    return ""


def _format_history(history: List[dict] | None) -> str:
    """Format conversation history into a readable string for the LLM."""
    if not history:
        return "No previous conversation history."

    formatted_turns = []
    for entry in history:
        turn_str = f"User: {entry.get('question', '')}\nAssistant: {entry.get('answer', '')}"
        formatted_turns.append(turn_str)

    return "\n\n".join(formatted_turns)


class SummarizationOutput(BaseModel):
    answer: str = Field(
        ...,
        description="The natural language answer to the user's question based on the context."
    )
    used_history: bool = Field(
        ...,
        description=(
            "CRITICAL: Set to True if the current question contains pronouns (e.g., 'he', 'she', 'it', 'that', 'they') "
            "or implicit references that refer back to the Conversation History. "
            "Set to False ONLY if the question is completely self-contained (e.g., 'What is HNSW?')."
        )
    )


# Define agents at module level for reuse
retrieval_agent = create_agent(
    model=create_chat_model(),
    tools=[retrieval_tool],
    system_prompt=RETRIEVAL_SYSTEM_PROMPT,
)

verification_agent = create_agent(
    model=create_chat_model(),
    tools=[],
    system_prompt=VERIFICATION_SYSTEM_PROMPT,
)

memory_summarization_agent = create_agent(
    model=create_chat_model(),
    tools=[],
    system_prompt=MEMORY_SUMMARIZATION_SYSTEM_PROMPT,
)

title_agent = create_agent(
    model=create_chat_model(),
    tools=[],
    system_prompt=TITLE_GENERATION_PROMPT,
)

summarization_llm = create_chat_model()
summarization_prompt = ChatPromptTemplate.from_messages([
    ("system", SUMMARIZATION_SYSTEM_PROMPT),
    ("human", "Question: {question}\n\nContext:\n{context}")
])
summarization_chain = summarization_prompt | summarization_llm.with_structured_output(SummarizationOutput)


def retrieval_node(state: QAState) -> QAState:
    """Retrieval Agent node: gathers context from vector store.

    This node:
    - Formats history for context-aware retrieval.
    - Sends the user's question + history to the Retrieval Agent.
    - The agent uses the attached retrieval tool to fetch document chunks.
    - Stores the consolidated context string in `state["context"]`.
    """
    question = state["question"]
    history_str = _format_history(state.get("history"))

    # We must pass 'question' and 'history' to fill the prompt variables
    result = retrieval_agent.invoke({
        "messages": [HumanMessage(content=question)],
        "question": question,
        "history": history_str
    })

    messages = result.get("messages", [])
    context = ""

    # Prefer the last ToolMessage content (from retrieval_tool)
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            context = str(msg.content)
            break

    return {
        "context": context,
    }


def summarization_node(state: QAState) -> QAState:
    """Summarization Agent node: generates draft answer from context.

    This node:
    - Sends question + context + history to the Summarization Agent.
    - Agent responds with a draft answer grounded in context and previous turns.
    - Stores the draft answer in `state["draft_answer"]`.
    """
    question = state["question"]
    context = state.get("context")
    history_list = state.get("history", []) or []
    history_str = _format_history(history_list)

    result: SummarizationOutput = summarization_chain.invoke({
        "history": history_str,
        "question": question,
        "context": context
    })

    used_history = result.used_history

    if not history_list:
        used_history = False

    return {
        "draft_answer": result.answer,
        "used_history": used_history
    }


def verification_node(state: QAState) -> QAState:
    """Verification Agent node: verifies and corrects the draft answer.

    This node:
    - Sends question + context + draft_answer to the Verification Agent.
    - Agent checks for hallucinations and unsupported claims.
    - Stores the final verified answer in `state["answer"]`.
    """
    question = state["question"]
    context = state.get("context", "")
    draft_answer = state.get("draft_answer", "")

    user_content = f"""Question: {question}

Context:
{context}

Draft Answer:
{draft_answer}

Please verify and correct the draft answer, removing any unsupported claims."""

    result = verification_agent.invoke(
        {"messages": [HumanMessage(content=user_content)]}
    )
    messages = result.get("messages", [])
    answer = _extract_last_ai_content(messages)

    return {
        "answer": answer,
    }


def memory_summarizer_node(state: QAState) -> QAState:
    history = state.get("history", []) or []

    if len(history) > 1:
        history_str = _format_history(history)
        user_content = f"Summarize this conversation history:\n\n{history_str}"

        result = memory_summarization_agent.invoke(
            {"messages": [HumanMessage(content=user_content)]}
        )
        messages = result.get("messages", [])
        summary = _extract_last_ai_content(messages)

        return {
            "conversation_summary": summary
        }

    return {}


def generate_chat_title(question: str, answer: str) -> str:
    prompt_content = f"Generate a title for this:\nQuestion: {question}\nAnswer: {answer}"

    result = title_agent.invoke({
        "messages": [HumanMessage(content=prompt_content)]
    })

    messages = result.get("messages", [])
    title = _extract_last_ai_content(messages).strip('"')

    if not title or len(title) > 50:
        return "New Chat"

    return title
