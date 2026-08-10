from fastapi import FastAPI, HTTPException
from langchain_core.messages import ToolMessage

from agent import agent
from schemas import (
    AgentQueryRequest,
    AgentQueryResponse,
    HistoryItem
)


# ============================================================
# 1. FastAPI Application
# ============================================================

app = FastAPI(
    title="Agentic RAG System",
    description=(
        "Agentic RAG system using Hybrid Retrieval, "
        "Wikipedia, DuckDuckGo and Joke API"
    ),
    version="1.0.0"
)


# ============================================================
# 2. System Prompt
# ============================================================

SYSTEM_PROMPT = """
You are an Agentic RAG assistant.

Select tools carefully based on the user's question.

AVAILABLE TOOLS

1. search_documents
Use ONLY when the user explicitly asks about the uploaded documents,
PDFs, document knowledge base, or information expected to be contained
in those documents.

2. search_wikipedia
Use for general factual and encyclopedic questions about people,
places, history, science, technology, organizations, and concepts.

3. search_duckduckgo
Use for current, recent, latest, today's, live, or time-sensitive
information.

4. get_joke
Use when the user asks for a joke.

TOOL ROUTING RULES

- General knowledge -> search_wikipedia
- Uploaded document/PDF question -> search_documents
- Current/latest/today/live information -> search_duckduckgo
- Joke request -> get_joke

- Do NOT search documents for an ordinary general-knowledge question.
- Do NOT use Wikipedia when only current information is requested.
- Do NOT call unrelated tools.
- Prefer one tool when one tool is sufficient.
- Use multiple tools ONLY when the user's question genuinely requires
  information from multiple sources.
- If the user explicitly asks for document information AND current
  information, use search_documents and search_duckduckgo.
- If a tool result is insufficient, another relevant tool may be used.
- Never invent tool names.
- Base factual answers on the tool results.
- If a tool fails, explain briefly or use another relevant tool.
- Always provide a final answer.
"""


# ============================================================
# 3. In-Memory History
# ============================================================

history: list[HistoryItem] = []


# ============================================================
# 4. Home Endpoint
# ============================================================

@app.get("/")
def home():

    return {
        "message": "Agentic RAG API is running"
    }


# ============================================================
# 5. POST /agent/query
# ============================================================

@app.post(
    "/agent/query",
    response_model=AgentQueryResponse
)
def query_agent(request: AgentQueryRequest):

    try:

        # ----------------------------------------------------
        # Send Query to Agent
        # ----------------------------------------------------

        response = agent.invoke(
            {
                "messages": [
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": request.query
                    }
                ]
            }
        )

        # ----------------------------------------------------
        # Get Messages
        # ----------------------------------------------------

        messages = response.get("messages", [])

        if not messages:
            raise ValueError(
                "Agent returned no messages."
            )

        # ----------------------------------------------------
        # Get Final Answer
        # ----------------------------------------------------

        answer = messages[-1].content

        if not answer:
            answer = (
                "The agent could not generate "
                "a final answer."
            )

        if not isinstance(answer, str):
            answer = str(answer)

        # ----------------------------------------------------
        # Find Tools Used
        # ----------------------------------------------------

        tools_used: list[str] = []

        for message in messages:

            tool_calls = getattr(
                message,
                "tool_calls",
                None
            )

            if not tool_calls:
                continue

            for tool_call in tool_calls:

                tool_name = tool_call.get("name")

                if (
                    tool_name
                    and tool_name not in tools_used
                ):
                    tools_used.append(tool_name)

        # ----------------------------------------------------
        # Find RAG Sources
        # ----------------------------------------------------

        sources: list[str] = []

        for message in messages:

            if not isinstance(message, ToolMessage):
                continue

            # Only extract sources from RAG tool
            if message.name != "search_documents":
                continue

            content = str(message.content)

            for line in content.splitlines():

                if line.startswith("Source:"):

                    source = line.replace(
                        "Source:",
                        "",
                        1
                    ).strip()

                    if (
                        source
                        and source not in sources
                    ):
                        sources.append(source)

        # ----------------------------------------------------
        # Create Response
        # ----------------------------------------------------

        result = AgentQueryResponse(
            answer=answer,
            tools_used=tools_used,
            sources=sources
        )

        # ----------------------------------------------------
        # Store History
        # ----------------------------------------------------

        history.append(
            HistoryItem(
                query=request.query,
                answer=answer,
                tools_used=tools_used,
                sources=sources
            )
        )

        # Keep only last 10 queries
        if len(history) > 10:
            del history[:-10]

        return result

    # --------------------------------------------------------
    # Error Handling
    # --------------------------------------------------------

    except Exception as e:

        print(f"Agent Error: {e}")

        raise HTTPException(
            status_code=500,
            detail=f"Agent failed: {str(e)}"
        )


# ============================================================
# 6. GET /agent/history
# ============================================================

@app.get(
    "/agent/history",
    response_model=list[HistoryItem]
)
def get_history():

    return list(reversed(history))