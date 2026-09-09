"""
MCP Server — exposes RAG retrieval as an MCP tool over HTTP.
Run this as its own long-lived process, separate from the agent/API.
"""

from fastmcp import FastMCP
from .retrieval import search_documents_raw
from DocumentChangeHandler import start_file_watcher

mcp = FastMCP("rag-knowledge-server")


@mcp.tool()
def search_knowledge_base(query: str) -> str:
    """
    Search the ingested technical manuals, text files, and PDFs and return
    the top reranked chunks with their sources.
    """
    return search_documents_raw(query)


if __name__ == "__main__":
    start_file_watcher()   # watches for new/changed docs, indexes incrementally
    mcp.run(transport="http", host="0.0.0.0", port=8001)