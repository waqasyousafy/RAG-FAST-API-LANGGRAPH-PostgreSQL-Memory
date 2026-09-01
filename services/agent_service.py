from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.postgres import PostgresSaver
from config import MODEL_NAME,DOC_DIRECTORY, DB_CONNECTION, COLLECTION_NAME
from database import get_db_connection
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document
load_dotenv()
from pathlib import Path
groqapikey=os.getenv("GROQ_API_KEY")


_db_vector = None
_agent_executor = None
_checkpointer = None

def initialize_vector_store():
    global _db_vector
    env_path = os.getenv('DIRECTORY', 'G:\\datato_ingestion')
    docDirectory = Path(env_path)
    if _db_vector is None:
        print("Initializing vector store on startup...")

        target_path = docDirectory
        print(f"Target document directory path: {target_path}")

        documents = []
        if not target_path.exists():
            print(f"Directory '{target_path}' not found. Creating it...")
            target_path.mkdir(parents=True, exist_ok=True)
            documents = [Document(page_content="Welcome! Please add your text or PDF files to this folder.", metadata={"source": "system_default"})]
        else:
            print("Scanning directory for files...")
            # Load all text files using your working pattern
            for txt_file in sorted(target_path.glob("*.txt")):
                try:
                    loader = TextLoader(str(txt_file), encoding="utf-8")
                    documents.extend(loader.load())
                    print(f"Loaded: {txt_file}")
                except Exception as e:
                    print(f"Warning: Could not load {txt_file}: {e}")
            
            # Load all PDF files using your working pattern
            for pdf_file in sorted(target_path.glob("*.pdf")):
                try:
                    loader = PyPDFLoader(str(pdf_file))
                    documents.extend(loader.load())
                    print(f"Loaded: {pdf_file}")
                except Exception as e:
                    print(f"Warning: Could not load {pdf_file}: {e}")

            if not documents:
                documents = [Document(page_content="Welcome! Please add your text or PDF files to this folder.", metadata={"source": "system_default"})]

        print(f"Total raw documents loaded: {len(documents)}")

        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        split_docs = text_splitter.split_documents(documents)
        print(f"Total text chunks after splitting: {len(split_docs)}")
        
        embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        _db_vector = PGVector.from_documents(
            documents=split_docs,
            embedding=embedding_function,
            collection_name=COLLECTION_NAME,
            connection=DB_CONNECTION,
            use_jsonb=True,
        )
        print("Vector store initialized successfully.")
    return _db_vector
@tool
def search_documents(query: str) -> str:
    """Search ingested technical manuals, text files, and PDFs for context to answer questions."""
    vector_store = initialize_vector_store()
    results = vector_store.max_marginal_relevance_search(query, k=3, fetch_k=5)
    if not results:
        return "No relevant documents found."
    formatted = [f"Source: {doc.metadata.get('source', 'Unknown')}\nContent: {doc.page_content}" for doc in results]
    return "\n\n".join(formatted)

from database import get_db_connection
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_core.messages import SystemMessage
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

def get_agent_executor():
    global _agent_executor, _checkpointer
    if _agent_executor is None:
        print("Initializing LangGraph ReAct Agent with Groq...")
        
        llm = ChatGroq(
            model_name=MODEL_NAME,
            api_key=groqapikey,
            temperature=0.1
        )
        # Define Johnson's system message instructions
        system_instruction = SystemMessage(
            content=(
                "You are Johnson, a helpful support desk and front-desk employee for our e-commerce store. "
                "Your duty is to assist customers strictly using the provided catalog data, order history tools, "
                "and internal feed data. "
                "CRITICAL RULE: If a user asks questions completely outside the provided feed data, store inventory, "
                "or order context, you must politely decline to answer and remind them that you can only assist with store-related inquiries. "
                "Never make up information or answer outside your designated support role."
            )
        )
        tools = [search_documents]
        
        # 1. Get database connection
        sync_conn = get_db_connection()
        
        # 2. CRITICAL: Enable autocommit to allow concurrent index creation
        sync_conn.autocommit = True
        
        # 3. Setup checkpointer
        _checkpointer = PostgresSaver(sync_conn)
        _checkpointer.setup()
        
        _agent_executor = create_react_agent(
            model=llm,
            tools=tools,
            checkpointer=_checkpointer,
            state_modifier=system_instruction
        )
        print("Agent executor ready.")
    return _agent_executor


def run_agent_workflow(query: str, user_id: str, thread_id: str) -> str:
    agent = get_agent_executor()
    # Composite configuration namespace mapping user and thread contexts safely
    config = {"configurable": {"thread_id": f"user_{user_id}_thread_{thread_id}"}}
    
    input_message = {"messages": [("user", query)]}
    output = agent.invoke(input_message, config=config)
    
    # Extract final message response from state messages
    return output["messages"][-1].content