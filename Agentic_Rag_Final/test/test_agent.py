from agent import agent


# ============================================================
# System Prompt
# ============================================================

SYSTEM_PROMPT = """
You are an Agentic RAG assistant.

Select the most appropriate tool based on the user's question.

AVAILABLE TOOLS

1. search_documents
Searches the uploaded local PDF documents.
Use this tool ONLY when the user explicitly asks about:
- uploaded documents
- PDFs
- my documents
- document knowledge base
- information from the provided documents

2. search_wikipedia
Searches Wikipedia.
Use this tool for general factual and encyclopedic questions about:
- people
- places
- languages
- history
- science
- technology
- organizations
- concepts

3. search_duckduckgo
Searches the web.
Use this tool for:
- current information
- latest information
- today's information
- recent events
- news
- current officials
- time-sensitive information

4. get_joke
Use this tool only when the user asks for a joke.

STRICT ROUTING RULES

- General knowledge -> search_wikipedia
- Uploaded PDF/document question -> search_documents
- Current/latest/today/news question -> search_duckduckgo
- Joke request -> get_joke

IMPORTANT:
- Prefer ONE tool when one tool is enough.
- Do NOT search uploaded documents for ordinary general knowledge.
- Do NOT use search_documents just because the documents might contain
  something remotely related to the question.
- Do NOT add document information unless the user asks for document-based information.
- Do NOT call unrelated tools.
- Use multiple tools ONLY when the user's question clearly requires multiple sources.
- Never invent tool names.
- Base the final answer on the selected tool results.
- Always provide a final answer.
"""


# ============================================================
# Start
# ============================================================

print("=" * 60)
print("Agentic RAG System Started")
print("Type 'exit' to quit")
print("=" * 60)


while True:

    query = input("\nYou : ").strip()

    # --------------------------------------------------------
    # Exit
    # --------------------------------------------------------

    if query.lower() == "exit":

        print("\nAgent stopped.")

        break

    # --------------------------------------------------------
    # Empty Input
    # --------------------------------------------------------

    if not query:

        print("Please enter a question.")

        continue

    try:

        # ----------------------------------------------------
        # Run Agent
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
                        "content": query
                    }
                ]
            }
        )

        # ----------------------------------------------------
        # Get Messages
        # ----------------------------------------------------

        messages = response.get(
            "messages",
            []
        )

        if not messages:

            print(
                "\nNo response received "
                "from the agent."
            )

            continue

        # ----------------------------------------------------
        # Final Answer
        # ----------------------------------------------------

        final_answer = messages[-1].content

        print("\nAssistant:\n")

        if final_answer:

            print(final_answer)

        else:

            print(
                "No final answer was generated."
            )

        # ----------------------------------------------------
        # Find Tools Used
        # ----------------------------------------------------

        tools_used = []

        for message in messages:

            tool_calls = getattr(
                message,
                "tool_calls",
                None
            )

            if not tool_calls:
                continue

            for tool_call in tool_calls:

                tool_name = tool_call.get(
                    "name"
                )

                if (
                    tool_name
                    and tool_name not in tools_used
                ):

                    tools_used.append(
                        tool_name
                    )

        # ----------------------------------------------------
        # Display Tools
        # ----------------------------------------------------

        if tools_used:

            print(
                "\nTools Used:",
                ", ".join(tools_used)
            )

        else:

            print(
                "\nTools Used: None"
            )

    # --------------------------------------------------------
    # Error
    # --------------------------------------------------------

    except Exception as e:

        print("\nAgent Error:")

        print(str(e))