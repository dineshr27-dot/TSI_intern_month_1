from dotenv import load_dotenv
from logger import logger

from langchain_core.messages import HumanMessage

from langsmith import traceable

from rag.loader import load_documents, split_documents
from rag.retriever import (
    create_vector_store,
    get_hybrid_retriever,
)

from tools.wikipedia_tool import (
    search_wikipedia as wikipedia_api,
)

from tools.duckduckgo_tool import (
    search_duckduckgo as duckduckgo_api,
)

from fine_tuning.qwen_chat_model import LocalQwenChatModel

from prompts.agent_prompt_v1 import (
    AGENT_PROMPT_V1,
    PROMPT_VERSION,
)


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# 2. LOAD DOCUMENTS
# ============================================================

documents = load_documents()

chunks = split_documents(
    documents
)

print(
    f"Pages Loaded: {len(documents)}"
)

print(
    f"Chunks Created: {len(chunks)}"
)


# ============================================================
# 3. CREATE VECTOR STORE + HYBRID RETRIEVER
# ============================================================

print(
    "\nCreating vector store..."
)

create_vector_store(
    chunks
)

print(
    "Creating hybrid retriever..."
)

retriever = get_hybrid_retriever(
    chunks
)


# ============================================================
# 4. DOCUMENT SEARCH
# ============================================================

@traceable(
    name="Search Documents",
    run_type="retriever",
)
def search_documents(
    query: str,
) -> str:

    logger.info(
        f"Tool called: search_documents | Query: {query}"
    )

    try:

        docs = retriever.invoke(
            query
        )[:3]

        if not docs:

            logger.warning(
                "search_documents returned no results"
            )

            return (
                "No relevant information was found "
                "in the uploaded documents."
            )

        results = []

        for i, doc in enumerate(
            docs,
            start=1,
        ):

            source = doc.metadata.get(
                "source",
                "Unknown source",
            )

            content = (
                doc.page_content
                .strip()
            )  

            results.append(
                f"Result {i}\n"
                f"Source: {source}\n"
                f"Content: {content}"
            )

        result = "\n\n".join(
            results
        )

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
# 5. WIKIPEDIA SEARCH
# ============================================================

@traceable(
    name="Search Wikipedia",
    run_type="tool",
)
def search_wikipedia(
    query: str,
) -> str:

    logger.info(
        f"Tool called: search_wikipedia | Query: {query}"
    )

    try:

        result = wikipedia_api(
            query
        )

        if not result:

            return (
                "No Wikipedia information was found "
                "for this query."
            )

        result = str(
            result
        )

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
# 6. DUCKDUCKGO SEARCH
# ============================================================

@traceable(
    name="Search DuckDuckGo",
    run_type="tool",
)
def search_duckduckgo(
    query: str,
) -> str:

    logger.info(
        f"Tool called: search_duckduckgo | Query: {query}"
    )

    try:

        result = duckduckgo_api(
            query
        )

        if not result:

            return (
                "No current web search results "
                "were found."
            )

        result = str(
            result
        )

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
# 7. MODEL LOADER
# ============================================================

def get_llm(
    model_type: str = "finetuned",
):

    """
    Load the requested Qwen model.

    model_type:
        base
            -> Qwen2.5-1.5B-Instruct

        finetuned
            -> Qwen2.5-1.5B-Instruct
               + QLoRA adapter
    """

    if model_type not in [
        "base",
        "finetuned",
    ]:

        raise ValueError(
            "model_type must be "
            "'base' or 'finetuned'"
        )

    if model_type == "base":

        logger.info(
            "Loading BASE Qwen model"
        )

    else:

        logger.info(
            "Loading FINE-TUNED Qwen model"
        )

    return LocalQwenChatModel(
        model_type=model_type
    )


# ============================================================
# 8. TOOL ROUTING
# ============================================================

@traceable(
    name="Agent Router",
    run_type="chain",
)
def route_query(
    query: str,
):

    """
    Select the correct tool for the query.

    Returns:
        tool_name, tool_result
    """

    query_lower = (
        query.lower()
    )

    # --------------------------------------------------------
    # SIMPLE CONVERSATION / GREETINGS
    # --------------------------------------------------------
    # Simple greetings should not trigger an external tool.

    greeting_patterns = [
        "hi",
        "hello",
        "hey",
        "hi there",
        "hello there",
        "hey there",
        "good morning",
        "good afternoon",
        "good evening",
    ]

    normalized_query = query_lower.strip(" .,!?;:")

    if normalized_query in greeting_patterns:

        logger.info(
            "Router selected: direct conversation"
        )

        return (
            "direct",
            "This is a simple conversational message. "
            "Respond naturally without using any external tool."
        )

    # --------------------------------------------------------
    # DOCUMENT / PDF
    # --------------------------------------------------------

    document_keywords = [
        "uploaded document",
        "uploaded pdf",
        "pdf",
        "document",
        "documents",
        "according to the document",
        "according to the pdf",
        "in the document",
        "in the pdf",
    ]

    if any(
        keyword in query_lower
        for keyword in document_keywords
    ):

        logger.info(
            "Router selected: search_documents"
        )

        result = search_documents(
            query
        )

        return (
            "search_documents",
            result,
        )

    # --------------------------------------------------------
    # CURRENT / LATEST
    # --------------------------------------------------------

    current_keywords = [
        "latest",
        "current",
        "today",
        "today's",
        "now",
        "recent",
        "recently",
        "live",
        "news",
        "this week",
        "this month",
    ]

    if any(
        keyword in query_lower
        for keyword in current_keywords
    ):

        logger.info(
            "Router selected: search_duckduckgo"
        )

        result = search_duckduckgo(
            query
        )

        return (
            "search_duckduckgo",
            result,
        )

    # --------------------------------------------------------
    # GENERAL KNOWLEDGE
    # --------------------------------------------------------

    logger.info(
        "Router selected: search_wikipedia"
    )

    result = search_wikipedia(
        query
    )

    return (
        "search_wikipedia",
        result,
    )


# ============================================================
# 9. BUILD VERSIONED PROMPT
# ============================================================

def build_answer_prompt(
    query: str,
    tool_name: str,
    tool_result: str,
) -> str:

    prompt = AGENT_PROMPT_V1.format(
        query=query,
        tool_name=tool_name,
        tool_result=tool_result,
    )

    logger.info(
        f"Using prompt version: {PROMPT_VERSION}"
    )

    return prompt


# ============================================================
# 10. GENERATE FINAL ANSWER
# ============================================================

@traceable(
    name="Qwen Generation",
    run_type="llm",
)
def generate_answer(
    query: str,
    tool_name: str,
    tool_result: str,
    model_type: str = "finetuned",
) -> str:

    prompt = build_answer_prompt(
        query=query,
        tool_name=tool_name,
        tool_result=tool_result,
    )

    # --------------------------------------------------------
    # Load selected model
    # --------------------------------------------------------

    selected_llm = get_llm(
        model_type
    )

    # --------------------------------------------------------
    # Generate
    # --------------------------------------------------------

    response = selected_llm.invoke(
        prompt
    )

    return response.content


# ============================================================
# 11. STREAM FINAL ANSWER
# ============================================================

def stream_agent_answer(
    query: str,
    model_type: str = "finetuned",
):

    """
    Stream the final answer from the selected Qwen model.

    model_type:
        base
        finetuned

    Events:
        token
        metadata
        done
    """

    logger.info(
        f"Streaming agent received query: {query}"
    )

    logger.info(
        f"Selected model: {model_type}"
    )

    # --------------------------------------------------------
    # ROUTE
    # --------------------------------------------------------

    tool_name, tool_result = (
        route_query(query)
    )

    logger.info(
        f"Streaming route selected: {tool_name}"
    )

    # --------------------------------------------------------
    # BUILD PROMPT
    # --------------------------------------------------------

    prompt = build_answer_prompt(
        query=query,
        tool_name=tool_name,
        tool_result=tool_result,
    )

    # --------------------------------------------------------
    # LOAD SELECTED QWEN
    # --------------------------------------------------------

    selected_llm = get_llm(
        model_type
    )

    # --------------------------------------------------------
    # STREAM QWEN
    # --------------------------------------------------------

    try:

        for chunk in selected_llm.stream_response(
            [
                HumanMessage(
                    content=prompt
                )
            ],
            max_new_tokens=512,
            temperature=0.7,
        ):

            if chunk:

                yield {
                    "type": "token",
                    "content": chunk,
                }

    except Exception as e:

        logger.exception(
            "Streaming generation failed"
        )

        yield {
            "type": "error",
            "message": str(e),
        }

        return

    # --------------------------------------------------------
    # MODEL NAME
    # --------------------------------------------------------

    if model_type == "base":

        model_name = (
            "Base Qwen "
            "(Qwen2.5-1.5B-Instruct)"
        )

    else:

        model_name = (
            "Fine-tuned Qwen "
            "(QLoRA)"
        )

    # --------------------------------------------------------
    # METADATA
    # --------------------------------------------------------

    yield {
        "type": "metadata",

        "tools_used": [
            tool_name
        ],

        "sources": extract_sources(
            tool_name,
            tool_result,
        ),

        "prompt_version": PROMPT_VERSION,

        "model_type": model_type,

        "model_name": model_name,
    }

    # --------------------------------------------------------
    # DONE
    # --------------------------------------------------------

    logger.info(
        f"Streaming agent completed | "
        f"Tool: {tool_name} | "
        f"Model: {model_type} | "
        f"Prompt: {PROMPT_VERSION}"
    )

    yield {
        "type": "done",
    }


# ============================================================
# 12. NORMAL AGENT
# ============================================================

@traceable(
    name="Agent Run",
    run_type="chain",
)
def run_agent(
    query: str,
    model_type: str = "finetuned",
):

    logger.info(
        f"Agent received query: {query}"
    )

    logger.info(
        f"Selected model: {model_type}"
    )

    # --------------------------------------------------------
    # ROUTE
    # --------------------------------------------------------

    tool_name, tool_result = (
        route_query(query)
    )

    # --------------------------------------------------------
    # GENERATE
    # --------------------------------------------------------

    answer = generate_answer(
        query=query,
        tool_name=tool_name,
        tool_result=tool_result,
        model_type=model_type,
    )

    logger.info(
        f"Agent completed | "
        f"Tool: {tool_name} | "
        f"Model: {model_type} | "
        f"Prompt: {PROMPT_VERSION}"
    )

    if model_type == "base":

        model_name = (
            "Base Qwen "
            "(Qwen2.5-1.5B-Instruct)"
        )

    else:

        model_name = (
            "Fine-tuned Qwen "
            "(QLoRA)"
        )

    return {
        "answer": answer,

        "tools_used": [
            tool_name
        ],

        "sources": extract_sources(
            tool_name,
            tool_result,
        ),

        "prompt_version": PROMPT_VERSION,

        "model_type": model_type,

        "model_name": model_name,
    }


# ============================================================
# 13. EXTRACT RAG SOURCES
# ============================================================

def extract_sources(
    tool_name: str,
    tool_result: str,
):

    sources = []

    if tool_name != "search_documents":

        return sources

    for line in tool_result.splitlines():

        if line.startswith(
            "Source:"
        ):

            source = line.replace(
                "Source:",
                "",
                1,
            ).strip()

            if (
                source
                and source not in sources
            ):

                sources.append(
                    source
                )

    return sources


# ============================================================
# 14. TEST AGENT
# ============================================================

if __name__ == "__main__":

    print(
        "\n" + "=" * 60
    )

    print(
        "AGENTIC RAG READY"
    )

    print(
        "=" * 60
    )

    print(
        f"Prompt Version: {PROMPT_VERSION}"
    )

    print(
        "\nAvailable models:"
    )

    print(
        "1. Base Qwen"
    )

    print(
        "2. Fine-tuned Qwen (QLoRA)"
    )

    while True:

        query = input(
            "\nEnter your question "
            "(type 'exit' to stop): "
        )

        if query.lower() == "exit":

            break

        try:

            model_choice = input(
                "Choose model "
                "(base/finetuned): "
            ).strip().lower()

            if model_choice not in [
                "base",
                "finetuned",
            ]:

                print(
                    "Invalid model. "
                    "Using finetuned."
                )

                model_choice = "finetuned"

            result = run_agent(
                query,
                model_type=model_choice,
            )

            print(
                "\n" + "=" * 60
            )

            print(
                "ANSWER"
            )

            print(
                "=" * 60
            )

            print(
                result["answer"]
            )

            print(
                "\nModel:"
            )

            print(
                result["model_name"]
            )

            print(
                "\nTool used:"
            )

            print(
                result["tools_used"]
            )

            print(
                "\nPrompt version:"
            )

            print(
                result["prompt_version"]
            )

            if result["sources"]:

                print(
                    "\nSources:"
                )

                for source in result[
                    "sources"
                ]:

                    print(
                        "-",
                        source,
                    )

        except Exception as e:

            print(
                f"\nAgent error: {e}"
            )