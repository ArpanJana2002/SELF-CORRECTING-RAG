"""
Ingestion pipeline: load raw documents (txt/md/pdf/urls), split them into
chunks, embed them, and persist them into a local Chroma vector store.

Run directly:
    python -m src.ingestion --path ./data
"""
import argparse
import glob
import os
import sys

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    TextLoader,
    UnstructuredMarkdownLoader,
    WebBaseLoader,
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config


def _load_local_files(path: str):
    """Load every .txt / .md file found under `path` (file or directory)."""
    docs = []
    if os.path.isdir(path):
        files = glob.glob(os.path.join(path, "**", "*"), recursive=True)
    else:
        files = [path]

    for f in files:
        if not os.path.isfile(f):
            continue
        ext = os.path.splitext(f)[1].lower()
        try:
            if ext == ".md":
                docs.extend(UnstructuredMarkdownLoader(f).load())
            elif ext == ".txt":
                docs.extend(TextLoader(f, encoding="utf-8").load())
            else:
                continue
            print(f"  loaded: {f}")
        except Exception as e:
            print(f"  [skip] failed to load {f}: {e}")
    return docs


def _load_urls(urls):
    if not urls:
        return []
    docs = WebBaseLoader(urls).load()
    for u in urls:
        print(f"  loaded url: {u}")
    return docs


def build_vectorstore(source_path: str = None, urls=None, reset: bool = False):
    """
    Build (or update) the Chroma vector store from local files and/or URLs.
    Returns the Chroma vectorstore instance.
    """
    if reset and os.path.isdir(config.CHROMA_PERSIST_DIR):
        import shutil

        shutil.rmtree(config.CHROMA_PERSIST_DIR)
        print(f"Cleared existing vector store at {config.CHROMA_PERSIST_DIR}")

    raw_docs = []
    if source_path:
        print(f"Loading local documents from: {source_path}")
        raw_docs.extend(_load_local_files(source_path))
    if urls:
        print(f"Loading {len(urls)} URL(s)...")
        raw_docs.extend(_load_urls(urls))

    if not raw_docs:
        raise ValueError("No documents were loaded. Check source_path/urls.")

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500, chunk_overlap=75
    )
    chunks = splitter.split_documents(raw_docs)
    print(f"Split {len(raw_docs)} document(s) into {len(chunks)} chunks.")

    embeddings = config.get_embeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=config.CHROMA_COLLECTION_NAME,
        persist_directory=config.CHROMA_PERSIST_DIR,
    )
    print(f"Persisted vector store to: {config.CHROMA_PERSIST_DIR}")
    return vectorstore


def load_vectorstore():
    """Load an already-built Chroma vector store from disk."""
    embeddings = config.get_embeddings()
    return Chroma(
        collection_name=config.CHROMA_COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=config.CHROMA_PERSIST_DIR,
    )


def get_retriever(k: int = None):
    vectorstore = load_vectorstore()
    return vectorstore.as_retriever(search_kwargs={"k": k or config.RETRIEVAL_K})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest documents into the vector store.")
    parser.add_argument("--path", type=str, default="./data", help="File or directory to ingest")
    parser.add_argument("--url", action="append", default=[], help="URL(s) to ingest (repeatable)")
    parser.add_argument("--reset", action="store_true", help="Wipe existing vector store first")
    args = parser.parse_args()

    build_vectorstore(source_path=args.path, urls=args.url, reset=args.reset)
    print("\nIngestion complete.")
