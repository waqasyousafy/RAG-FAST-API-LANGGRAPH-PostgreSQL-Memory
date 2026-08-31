import time
import asyncio
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
import os

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
        # Wait a second to ensure file write operations have fully completed
        await asyncio.sleep(1)
        
        from services.agent_service import initialize_vector_store
        vector_store = initialize_vector_store()
        
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
                text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
                split_docs = text_splitter.split_documents(documents)
                
                # Optional: If you want to replace old chunks for this file, 
                # you can filter/delete matching metadata source first if supported,
                # otherwise add_documents appends new vectors cleanly.
                vector_store.add_documents(split_docs)
                print(f"[Watcher] Successfully indexed updates for: {path.name}")
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