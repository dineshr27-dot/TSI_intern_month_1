from dotenv import load_dotenv
from logger import logger

from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent

from rag.loader import load_documents, split_documents
from rag.retriever import create_vector_store, get_hybrid_retriever

from tools.wikipedia_tool import search_wikipedia as wikipedia_api
from tools.duckduckgo_tool import search_duckduckgo as duckduckgo_api
from tools.joke_tool import get_joke as joke_api


# ============================================================
# 1. Load Environment Variables
# ============================================================

load_dotenv()


# ============================================================
# 2. Load Documents
# ============================================================

documents = load_documents()
chunks = split_documents(documents)

print(f"Pages Loaded: {len(documents)}")
print(f"Chunks Created: {len(chunks)}")


# ============================================================
# 3. Create Vector Store + Hybrid Retriever
# ============================================================

# Create/update ChromaDB
create_vector_store(chunks)

# BM25 + ChromaDB
retriever = get_hybrid_retriever(chunks)


# ============================================================
# 4. Document RAG Tool
# ============================================================

@tool
def search_documents(query: str) -> str:
    """
    Search the uploaded PDF documents using hybrid retrieval.

    Use this tool when the user asks for information specifically
    from uploaded documents, PDFs, or the local document knowledge base.

    Args:
        query: Question or search query for the uploaded documents.

    Returns:
        Relevant document content with source information.
    """

    logger.info(
        f"Tool called: search_documents | Query: {query}"
    )

    try:
        docs = retriever.invoke(query)[:3]

        if not docs:
            logger.warning(
                "search_documents returned no results"
            )

            return (
                "No relevant information was found "
                "in the uploaded documents."
            )

        results = []

        for i, doc in enumerate(docs, start=1):

            source = doc.metadata.get(
                "source",
                "Unknown source"
            )

            content = doc.page_content.strip()

            results.append(
                f"Result {i}\n"
                f"Source: {source}\n"
                f"Content: {content}"
            )

        result = "\n\n".join(results)

        logger.info(
            f"Tool response: search_documents | "
            f"{result[:300]}"
        )

        return result

    except Exception as e:

        logger.exception(
            "search_documents failed"
        )

        return (
            "Document search failed. "
            f"Error: {str(e)}"
        )


# ============================================================
# 5. Wikipedia Tool
# ============================================================

@tool
def search_wikipedia(query: str) -> str:
    """
    Search Wikipedia for general factual and encyclopedic information.

    Use this tool for general knowledge about people, places,
    history, science, technology, organizations, and concepts.

    Do not use this tool when current or latest information is required.

    Args:
        query: Topic to search on Wikipedia.

    Returns:
        Relevant Wikipedia information.
    """

    logger.info(
        f"Tool called: search_wikipedia | Query: {query}"
    )

    try:
        result = wikipedia_api(query)

        if not result:

            logger.warning(
                "search_wikipedia returned no results"
            )

            return (
                "No Wikipedia information was found "
                "for this query."
            )

        result = str(result)

        logger.info(
            f"Tool response: search_wikipedia | "
            f"{result[:300]}"
        )

        return result

    except Exception as e:

        logger.exception(
            "search_wikipedia failed"
        )

        return (
            "Wikipedia search failed. "
            f"Error: {str(e)}"
        )


# ============================================================
# 6. DuckDuckGo Tool
# ============================================================

@tool
def search_duckduckgo(query: str) -> str:
    """
    Search the web using DuckDuckGo.

    Use this tool for current, latest, recent, today's,
    news-related, or other time-sensitive information.

    Args:
        query: Web search query.

    Returns:
        Relevant web search results.
    """

    logger.info(
        f"Tool called: search_duckduckgo | Query: {query}"
    )

    try:
        result = duckduckgo_api(query)

        if not result:

            logger.warning(
                "search_duckduckgo returned no results"
            )

            return (
                "No current web search results "
                "were found."
            )

        result = str(result)

        logger.info(
            f"Tool response: search_duckduckgo | "
            f"{result[:300]}"
        )

        return result

    except Exception as e:

        logger.exception(
            "search_duckduckgo failed"
        )

        return (
            "DuckDuckGo search failed. "
            f"Error: {str(e)}"
        )


# ============================================================
# 7. Joke Tool
# ============================================================

@tool
def get_joke() -> str:
    """
    Get one random joke.

    Use this tool only when the user asks for a joke.

    Returns:
        One random joke.
    """

    logger.info(
        "Tool called: get_joke"
    )

    try:
        result = joke_api()

        if not result:

            logger.warning(
                "get_joke returned no result"
            )

            return "No joke was returned."

        result = str(result)

        logger.info(
            f"Tool response: get_joke | "
            f"{result[:300]}"
        )

        return result

    except Exception as e:

        logger.exception(
            "get_joke failed"
        )

        return (
            "Joke API failed. "
            f"Error: {str(e)}"
        )


# ============================================================
# 8. Groq LLM
# ============================================================

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=2048,
    timeout=30,
    max_retries=2
)


# ============================================================
# 9. Register Tools
# ============================================================

tools = [
    search_documents,
    search_wikipedia,
    search_duckduckgo,
    get_joke
]


# ============================================================
# 10. Create Agent
# ============================================================

agent = create_react_agent(
    model=llm,
    tools=tools
)