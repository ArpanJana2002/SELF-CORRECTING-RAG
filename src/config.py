"""
Central configuration for the self-correcting RAG agent.
Everything is driven by environment variables (see .env.example).
"""
import os
from dotenv import load_dotenv

load_dotenv()


def _get_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "y")


# --- LLM ---
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic").lower()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# --- Embeddings ---
EMBEDDINGS_PROVIDER = os.getenv("EMBEDDINGS_PROVIDER", "local").lower()
LOCAL_EMBEDDINGS_MODEL = os.getenv(
    "LOCAL_EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

# --- Web search (corrective fallback) ---
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# --- Vector store ---
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./vectorstore")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "self_correcting_rag")

# --- Retrieval / correction tuning ---
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "4"))
MAX_CORRECTION_LOOPS = int(os.getenv("MAX_CORRECTION_LOOPS", "3"))


def get_llm(temperature: float = 0.0):
    """Return a chat model instance based on LLM_PROVIDER."""
    if LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI

        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is not set but LLM_PROVIDER=openai")
        return ChatOpenAI(model=OPENAI_MODEL, temperature=temperature, api_key=OPENAI_API_KEY)

    # default: anthropic
    from langchain_anthropic import ChatAnthropic

    if not ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY is not set but LLM_PROVIDER=anthropic")
    return ChatAnthropic(model=ANTHROPIC_MODEL, temperature=temperature, api_key=ANTHROPIC_API_KEY)


def get_embeddings():
    """Return an embeddings instance based on EMBEDDINGS_PROVIDER."""
    if EMBEDDINGS_PROVIDER == "openai":
        from langchain_openai import OpenAIEmbeddings

        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is not set but EMBEDDINGS_PROVIDER=openai")
        return OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)

    # default: local, free, no API key
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=LOCAL_EMBEDDINGS_MODEL)
