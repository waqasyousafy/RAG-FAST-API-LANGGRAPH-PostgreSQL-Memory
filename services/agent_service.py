from logging import config

from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.postgres import PostgresSaver
from config import MODEL_NAME, DOC_DIRECTORY, DB_CONNECTION, COLLECTION_NAME
from database import get_db_connection
from langchain_groq import ChatGroq
from langchain_deepseek import ChatDeepSeek
from dotenv import load_dotenv
import os

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from pathlib import Path
from langsmith import traceable
from langchain_core.messages.utils import trim_messages, count_tokens_approximately

from cache import ResponseCache
from rate_limiter import RateLimiter
from security import SecurityPipeline

load_dotenv()
from config import get_settings

groqapikey = os.getenv("GROQ_API_KEY")
setting = get_settings()

_db_vector = None
_agent_executor = None
_fallback_agent_executor = None
_checkpointer = None
_reranker = None

# --- New layers, built once at module level ---
_response_cache = ResponseCache(
    redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
    ttl_seconds=3600,  # 1 hour — pair with _response_cache.flush() after re-ingestion
)
_rate_limiter = RateLimiter(
    redis_client=_response_cache._client,  # reuse the same Redis connection
    max_requests=20,
    window_seconds=60,
)
_security = SecurityPipeline()


def get_reranker():
    global _reranker
    if _reranker is None:
        print("Loading reranker model...")
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker


@traceable(name="ingest_and_build_vector_store", run_type="chain")
def initialize_vector_store():
    global _db_vector

    env_path = os.getenv('DIRECTORY') or str(DOC_DIRECTORY)
    docDirectory = Path(env_path)

    if _db_vector is None:
        print("Initializing vector store on startup...")
        target_path = docDirectory
        print(f"Target document directory path: {target_path}")

        documents = []
        if not target_path.exists():
            print(f"Directory '{target_path}' not found. Creating it...")
            target_path.mkdir(parents=True, exist_ok=True)
            documents = [Document(
                page_content="Welcome! Please add your text or PDF files to this folder.",
                metadata={"source": "system_default"}
            )]
        else:
            print("Scanning directory for files...")
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

            if not documents:
                print("WARNING: No .txt or .pdf files found — using placeholder document.")
                documents = [Document(
                    page_content="Welcome! Please add your text or PDF files to this folder.",
                    metadata={"source": "system_default"}
                )]

        print(f"Total raw documents loaded: {len(documents)}")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=900,
            chunk_overlap=150,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
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

        # Since re-ingestion can change answers, don't let old cached
        # responses survive on TTL alone — flush explicitly.
        _response_cache.flush()
        print("Response cache flushed after (re-)ingestion.")

    return _db_vector


@traceable(name="vector_retrieval", run_type="retriever")
def retrieve_candidates(query: str, k: int = 3, fetch_k: int = 5):
    vector_store = initialize_vector_store()
    results = vector_store.max_marginal_relevance_search(query, k=k, fetch_k=fetch_k)
    print(f"[retrieve_candidates] query={query!r} -> {len(results)} raw hits")
    for r in results:
        print(f"  - {r.metadata.get('source', 'Unknown')}: {r.page_content[:100]!r}")
    return results


@traceable(name="cross_encoder_rerank", run_type="chain")
def rerank_documents(query: str, documents, top_n: int = 3):
    if not documents:
        return []
    reranker = get_reranker()
    pairs = [[query, doc.page_content] for doc in documents]
    scores = reranker.predict(pairs)
    scored_docs = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
    print(f"[rerank_documents] scores={[float(s) for s, _ in scored_docs]}")
    return [doc for score, doc in scored_docs[:top_n]]


@tool
@traceable(name="search_documents_tool", run_type="tool")
def search_documents(query: str) -> str:
    """Search ingested technical manuals, text files, and PDFs for context to answer questions."""
    results = retrieve_candidates(query, k=3, fetch_k=5)
    if not results:
        return "No relevant documents found."

    top_results = rerank_documents(query, results, top_n=3)

    MAX_CHARS_PER_CHUNK = 500
    formatted = [
        f"Source: {doc.metadata.get('source', 'Unknown')}\n"
        f"Content: {doc.page_content[:MAX_CHARS_PER_CHUNK]}"
        for doc in top_results
    ]
    return "\n\n".join(formatted)


SYSTEM_INSTRUCTION = (
    "You are Johnson, a researcher. For every user question, you MUST call the "
    "search_documents tool first to look for relevant context before answering. "
    "Only answer using information returned by that tool. If the tool returns "
    "'No relevant documents found.' or nothing that actually answers the question, "
    "say you don't know — do not answer from your own general knowledge."
)


def pre_model_hook(state):
    trimmed = trim_messages(
        state["messages"],
        strategy="last",
        token_counter=count_tokens_approximately,
        max_tokens=3000,
        start_on="human",
        include_system=True,
    )
    return {"llm_input_messages": trimmed}


def get_agent_executor():
    global _agent_executor, _checkpointer
    if _agent_executor is None:
        print("Initializing LangGraph ReAct Agent with Groq...")

        primary_llm = ChatGroq(
            model_name=setting.primary_model,
            api_key=groqapikey,
            temperature=0.1,
            max_retries=0,
            max_tokens=500,
        )

        tools = [search_documents]

        sync_conn = get_db_connection()
        sync_conn.autocommit = True
        _checkpointer = PostgresSaver(sync_conn)
        _checkpointer.setup()

        _agent_executor = create_react_agent(
            model=primary_llm,
            tools=tools,
            prompt=SYSTEM_INSTRUCTION,
            pre_model_hook=pre_model_hook,
            checkpointer=_checkpointer,
        )
        print("Agent executor ready.")
    return _agent_executor


def get_fallback_agent_executor():
    global _fallback_agent_executor, _checkpointer
    if _fallback_agent_executor is None:
        fallback_llm = ChatGroq(
            model_name=setting.fallback_model,
            api_key=setting.groq_api_fallback,
            temperature=0.1,
            max_retries=2,
            max_tokens=500,
        )

        tools = [search_documents]

        _fallback_agent_executor = create_react_agent(
            model=fallback_llm,
            tools=tools,
            prompt=SYSTEM_INSTRUCTION,
            pre_model_hook=pre_model_hook,
            checkpointer=_checkpointer,
        )
        print("Fallback agent executor ready.")
    return _fallback_agent_executor


def _log_token_usage(label: str, result: dict):
    try:
        last_msg = result["messages"][-1]
        usage = getattr(last_msg, "usage_metadata", None)
        if usage is None:
            usage = last_msg.response_metadata.get("token_usage")
        print(f"[tokens:{label}] {usage}")
    except Exception:
        pass


@traceable(name="run_agent_workflow", run_type="chain")
def run_agent_workflow(query: str, user_id: str, thread_id: str) -> str:
    # 1. Rate limit — cheapest check, reject before anything else runs
    allowed, info = _rate_limiter.is_allowed(user_id)
    if not allowed:
        return f"You're sending requests too quickly. Please wait {info['reset_in']}s and try again."

    # 2. Input security — injection check + PII masking
    is_allowed, cleaned_query, notes = _security.check_input(query)
    if not is_allowed:
        print(f"[security] blocked input for user={user_id}: {notes}")
        return "I can't process that request."
    if notes:
        print(f"[security] input notes for user={user_id}: {notes}")

    # 3. Cache check — keyed on the cleaned query
    cached = _response_cache.get(cleaned_query)
    if cached is not None:
        print("[cache] hit — skipping LLM call entirely")
        return cached

    # 4. Run the agent (primary, fallback on failure)
    agent = get_agent_executor()
    agent_config = {
        "configurable": {"thread_id": f"user_{user_id}_thread_{thread_id}"},
        "metadata": {"user_id": user_id},
    }
    input_message = {"messages": [("user", cleaned_query)]}

    try:
        output = agent.invoke(input_message, config=agent_config)
        _log_token_usage("primary", output)
    except Exception as e:
        print(f"[Agent Error] {e}. Attempting fallback agent...")
        fallback_agent = get_fallback_agent_executor()
        output = fallback_agent.invoke(input_message, config=agent_config)
        _log_token_usage("fallback", output)

    raw_answer = output["messages"][-1].content

    # 5. Output security — mask PII leakage, block harmful content
    answer, warnings = _security.check_output(raw_answer)
    if warnings:
        print(f"[security] output warnings for user={user_id}: {warnings}")

    # 6. Grounding check — log if the agent never actually searched docs
    tool_was_called = any(
        getattr(m, "name", None) == "search_documents" for m in output["messages"]
    )
    if not tool_was_called:
        print(f"[guardrail] answered without retrieval for query: {cleaned_query!r}")

    # 7. Cache the final (security-checked) answer
    _response_cache.set(cleaned_query, answer)

    return answer