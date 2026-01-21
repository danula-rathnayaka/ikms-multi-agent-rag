"""Vector store wrapper for Pinecone integration with LangChain."""

from functools import lru_cache
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone

from ..config import get_settings


@lru_cache(maxsize=1)
def _get_vector_store() -> PineconeVectorStore:
    """Create a PineconeVectorStore instance configured from settings."""
    settings = get_settings()

    pc = Pinecone(api_key=settings.pinecone_api_key)
    index = pc.Index(settings.pinecone_index_name)

    embeddings = OpenAIEmbeddings(
        model=settings.openai_embedding_model_name,
        api_key=settings.openai_api_key,
    )

    return PineconeVectorStore(
        index=index,
        embedding=embeddings,
    )


def get_retriever(k: int | None = None):
    """Get a Pinecone retriever instance.

    Args:
        k: Number of documents to retrieve (defaults to config value).

    Returns:
        PineconeVectorStore instance configured as a retriever.
    """
    settings = get_settings()
    if k is None:
        k = settings.retrieval_k

    vector_store = _get_vector_store()
    return vector_store.as_retriever(search_kwargs={"k": k})


def retrieve(query: str, k: int | None = None) -> List[Document]:
    """Retrieve documents from Pinecone for a given query.

    Args:
        query: Search query string.
        k: Number of documents to retrieve (defaults to config value).

    Returns:
        List of Document objects with metadata (including page numbers).
    """
    retriever = get_retriever(k=k)
    return retriever.invoke(query)


def index_documents(file_path: Path) -> int:
    """Index a list of Document objects into the Pinecone vector store.

    Args:
        docs: Documents to embed and upsert into the vector index.

    Returns:
        The number of documents indexed.
    """
    loader = PyPDFLoader(str(file_path), mode="single")
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(docs)

    vector_store = _get_vector_store()
    vector_store.add_documents(texts)
    return len(texts)


def delete_document_vectors(file_path: Path) -> bool:
    try:
        settings = get_settings()
        pc = Pinecone(api_key=settings.pinecone_api_key)
        index = pc.Index(settings.pinecone_index_name)
        source_id = str(file_path)

        index.delete(filter={"source": {"$eq": source_id}})
        return True
    except Exception as e:
        print(f"Error deleting vectors for {file_path}: {e}")
        return False


def delete_all_vectors() -> bool:
    """Wipe the entire Pinecone index. Used on server startup."""
    try:
        settings = get_settings()
        pc = Pinecone(api_key=settings.pinecone_api_key)
        index = pc.Index(settings.pinecone_index_name)

        # This deletes every single vector in the namespace
        index.delete(delete_all=True)
        print("Pinecone Index Wiped Successfully.")
        return True
    except Exception as e:
        print(f"Error wiping Pinecone index: {e}")
        return False
