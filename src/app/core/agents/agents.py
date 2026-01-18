"""Agent implementations for the multi-agent RAG flow.

This module defines three LangChain agents (Retrieval, Summarization,
Verification) and thin node functions that LangGraph uses to invoke them.
"""

from typing import List

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from .prompts import (
    RETRIEVAL_SYSTEM_PROMPT,
    SUMMARIZATION_SYSTEM_PROMPT,
    VERIFICATION_SYSTEM_PROMPT,
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


# Define agents at module level for reuse
retrieval_agent = create_agent(
    model=create_chat_model(),
    tools=[retrieval_tool],
    system_prompt=RETRIEVAL_SYSTEM_PROMPT,
)

summarization_agent = create_agent(
    model=create_chat_model(),
    tools=[],
    system_prompt=SUMMARIZATION_SYSTEM_PROMPT,
)

verification_agent = create_agent(
    model=create_chat_model(),
    tools=[],
    system_prompt=VERIFICATION_SYSTEM_PROMPT,
)


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
    history_str = _format_history(state.get("history"))

    user_content = f"Question: {question}\n\nContext:\n{context}"

    # We must pass 'history', 'question', and 'context' to fill the prompt variables
    result = summarization_agent.invoke({
        "messages": [HumanMessage(content=user_content)],
        "question": question,
        "context": context,
        "history": history_str
    })

    messages = result.get("messages", [])
    draft_answer = _extract_last_ai_content(messages)

    return {
        "draft_answer": draft_answer,
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
