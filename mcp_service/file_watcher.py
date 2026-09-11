"""
file_watcher.py — watches the ingestion directory and incrementally indexes
new/changed documents. Owned by the retrieval side (MCP server), not the
main API app — this process is the only one that should touch the vector store.
"""

import os
import asyncio
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .retrieval import get_vector_store
from cache import ResponseCache

_cache_for_invalidation = ResponseCache(
    redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
    ttl_seconds=3600,
)

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150


class DocumentChangeHandler(FileSystemEventHandler):
    def __init__(self, loop):
        self.loop = loop

    def on_created(self, event):
        if event.is_directory:
            return
        print(f"\n[Watcher] New file added: {event.src_path}")
        asyncio.run_coroutine_threadsafe(self.process_file(event.src_path), self.loop)

    def on_modified(self, event):
        if event.is_directory:
            return
        print(f"\n[Watcher] File modified: {event.src_path}")
        asyncio.run_coroutine_threadsafe(self.process_file(event.src_path), self.loop)

    async def process_file(self, file_path: str):
        await asyncio.sleep(1)

        vector_store = get_vector_store()
        path = Path(file_path)
        documents = []

        try:
            if path.suffix.lower() == ".txt":
                loader = TextLoader(str(path), encoding="utf-8")
                documents = loader.load()
            elif path.suffix.lower() == ".pdf":
                loader = PyPDFLoader(str(path))
                documents = loader.load()

            if documents:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=CHUNK_SIZE,
                    chunk_overlap=CHUNK_OVERLAP,
                    separators=["\n\n", "\n", ". ", " ", ""],
                )
                split_docs = text_splitter.split_documents(documents)
                vector_store.add_documents(split_docs)
                print(f"[Watcher] Indexed {len(split_docs)} chunk(s) for: {path.name}")

                _cache_for_invalidation.flush()
                print("[Watcher] Response cache flushed after document update.")

        except Exception as e:
            print(f"[Watcher Error] Failed to process {path.name}: {e}")


def start_file_watcher():
    loop = asyncio.get_event_loop()
    event_handler = DocumentChangeHandler(loop)
    observer = Observer()
    target_path = os.getenv('DIRECTORY', 'G:\\datato_ingestion')

    observer.schedule(event_handler, path=target_path, recursive=False)
    observer.start()
    print(f"File watcher successfully started on directory: {target_path}")