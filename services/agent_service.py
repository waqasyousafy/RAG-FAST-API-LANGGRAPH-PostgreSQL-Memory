"""
agent.py — LLM orchestration + guardrails. Connects to a separately-running
MCP server (mcp_server.py) for retrieval instead of doing retrieval locally.
Async throughout so it's safe to call from FastAPI's lifespan and routes.
"""

import os
from dotenv import load_dotenv

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq
from langchain_core.messages.utils import trim_messages, count_tokens_approximately
from langsmith import traceable

from cache import ResponseCache
from rate_limiter import RateLimiter
from security import SecurityPipeline
from config import get_settings

load_dotenv()

groqapikey = os.getenv("GROQ_API_KEY")
setting = get_settings()

_agent_executor = None
_fallback_agent_executor = None
_mcp_tools = None

MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8001/mcp")

_mcp_client = MultiServerMCPClient({
    "rag": {
        "transport": "http",
        "url": MCP_SERVER_URL,
    }
})

_response_cache = ResponseCache(
    redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
    ttl_seconds=3600,
)
_rate_limiter = RateLimiter(
    redis_client=_response_cache._client,
    max_requests=20,
    window_seconds=60,
)
_security = SecurityPipeline()


async def get_mcp_tools():
    """
    Fetch tools from the MCP server once and cache them. Requires
    mcp_server.py to already be running and reachable at MCP_SERVER_URL.
    """
    global _mcp_tools
    if _mcp_tools is None:
        print(f"Connecting to MCP server at {MCP_SERVER_URL} ...")
        _mcp_tools = await _mcp_client.get_tools()
        print(f"Loaded tool(s) from MCP server: {[t.name for t in _mcp_tools]}")
    return _mcp_tools


SYSTEM_INSTRUCTION = (
    "You are Johnson, a researcher. For every user question, you MUST call the "
    "search_knowledge_base tool first to look for relevant context before answering. "
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


async def get_agent_executor():
    global _agent_executor
    if _agent_executor is None:
        print("Initializing LangGraph ReAct Agent with Groq...")

        primary_llm = ChatGroq(
            model_name=setting.primary_model,
            api_key=groqapikey,
            temperature=0.1,
            max_retries=0,
            max_tokens=500,
        )

        tools = await get_mcp_tools()

        _agent_executor = create_react_agent(
            model=primary_llm,
            tools=tools,
            prompt=SYSTEM_INSTRUCTION
        )
        print("Agent executor ready.")
    return _agent_executor


async def get_fallback_agent_executor():
    global _fallback_agent_executor
    if _fallback_agent_executor is None:
        fallback_llm = ChatGroq(
            model_name=setting.fallback_model,
            api_key=setting.groq_api_fallback,
            temperature=0.1,
            max_retries=2,
            max_tokens=500,
        )

        tools = await get_mcp_tools()

        _fallback_agent_executor = create_react_agent(
            model=fallback_llm,
            tools=tools,
            prompt=SYSTEM_INSTRUCTION
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
async def run_agent_workflow(query: str, user_id: str) -> str:
    # 1. Rate limit
    allowed, info = _rate_limiter.is_allowed(user_id)
    if not allowed:
        return f"You're sending requests too quickly. Please wait {info['reset_in']}s and try again."

    # 2. Input security
    is_allowed, cleaned_query, notes = _security.check_input(query)
    if not is_allowed:
        print(f"[security] blocked input for user={user_id}: {notes}")
        return "I can't process that request."
    if notes:
        print(f"[security] input notes for user={user_id}: {notes}")

    # 3. Cache check
    cached = _response_cache.get(cleaned_query)
    if cached is not None:
        print("[cache] hit — skipping LLM call entirely")
        return cached

    # 4. Run the agent (primary, fallback on failure) — async throughout
    agent = await get_agent_executor()
    input_message = {"messages": [("user", cleaned_query)]}

    try:
        output = await agent.ainvoke(input_message)
        _log_token_usage("primary", output)
    except Exception as e:
        print(f"[Agent Error] {e}. Attempting fallback agent...")
        try:
            fallback_agent = await get_fallback_agent_executor()
            output = await fallback_agent.ainvoke(input_message)
            _log_token_usage("fallback", output)
        except Exception as e2:
            print(f"[Fallback Agent Error] {e2}")
            return "I'm having trouble processing your request right now. Please try again shortly."

    raw_answer = output["messages"][-1].content

    # 5. Output security
    answer, warnings = _security.check_output(raw_answer)
    if warnings:
        print(f"[security] output warnings for user={user_id}: {warnings}")

    # 6. Grounding check
    tool_was_called = any(
        getattr(m, "name", None) == "search_knowledge_base" for m in output["messages"]
    )
    if not tool_was_called:
        print(f"[guardrail] answered without retrieval for query: {cleaned_query!r}")

    # 7. Cache the final answer
    _response_cache.set(cleaned_query, answer)

    return answer