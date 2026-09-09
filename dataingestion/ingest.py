"""
One-time (or on-demand) document ingestion.
Run this manually whenever documents in DOC_DIRECTORY change:
    python ingest.py
Do NOT call this from the server's startup path — the server should only
attach to what this script already built.
"""

import os
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from config import DOC_DIRECTORY, DB_CONNECTION, COLLECTION_NAME


def load_documents() -> list[Document]:
    env_path = os.getenv('DIRECTORY') or str(DOC_DIRECTORY)
    target_path = Path(env_path)

    documents = []
    if not target_path.exists():
        target_path.mkdir(parents=True, exist_ok=True)
        print(f"Directory '{target_path}' created — add files and re-run.")
        return documents

    for txt_file in sorted(target_path.glob("*.txt")):
        try:
            loader = TextLoader(str(txt_file), encoding="utf-8")
            documents.extend(loader.load())
            print(f"Loaded: {txt_file}")
        except Exception as e:
            print(f"Warning: Could not load {txt_file}: {e}")

    for pdf_file in sorted(target_path.glob("*.pdf")):
        try:
            loader = PyPDFLoader(str(pdf_file))
            documents.extend(loader.load())
            print(f"Loaded: {pdf_file}")
        except Exception as e:
            print(f"Warning: Could not load {pdf_file}: {e}")

    return documents


def main():
    documents = load_documents()
    if not documents:
        print("No documents found — nothing to ingest.")
        return

    print(f"Total raw documents loaded: {len(documents)}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900, chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    split_docs = text_splitter.split_documents(documents)
    print(f"Total text chunks after splitting: {len(split_docs)}")

    print("Loading embedding model...")
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print("Embedding and writing to PGVector (this is the slow part — expected here, not on server start)...")
    PGVector.from_documents(
        documents=split_docs,
        embedding=embedding_function,
        collection_name=COLLECTION_NAME,
        connection=DB_CONNECTION,
        use_jsonb=True,
    )
    print(f"Ingestion complete: {len(split_docs)} chunks written to collection '{COLLECTION_NAME}'.")


if __name__ == "__main__":
    main()