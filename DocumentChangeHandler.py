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

from mcp_service.retrieval import get_vector_store  # was: services.agent_service.initialize_vector_store
from cache import ResponseCache

# Same Redis the main API's cache uses — flushing here invalidates it there too,
# since both processes point at the same Redis instance.
_cache_for_invalidation = ResponseCache(
    redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
    ttl_seconds=3600,
)

# Match ingest.py / retrieval.py so chunking is consistent regardless of
# whether a document arrived at initial ingestion or via the live watcher.
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
        # Let the write finish before reading — quick files can still be
        # mid-write when the event fires.
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

                # NOTE: this only ADDS chunks. If this is a modified file
                # (not new), the file's OLD chunks stay indexed alongside
                # the new ones — known limitation, not fixed here. If you
                # need clean replacement, this needs a delete-by-source-
                # metadata step before add_documents, which depends on
                # your PGVector version's filter support — worth checking
                # before relying on this for frequently-edited documents.
                vector_store.add_documents(split_docs)
                print(f"[Watcher] Indexed {len(split_docs)} chunk(s) for: {path.name}")

                # Invalidate cached answers, since new content may change
                # what a previously-cached query should now return.
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