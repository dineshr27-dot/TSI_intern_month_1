from agent import agent


SYSTEM_PROMPT = """
You are an Agentic RAG assistant.

Choose the most appropriate available tool based on the user's query.

Available tools:

- search_documents: Search uploaded PDF documents.
- search_wikipedia: Search general factual information.
- search_duckduckgo: Search current or recent web information.
- get_joke: Get a random joke.

Rules:

- Choose the tool based on the user's intent.
- Do not call unrelated tools.
- Prefer one tool when one tool is enough.
- Use multiple tools only when necessary.
- Always provide a final answer.
"""


print("=" * 60)
print("Ambiguous Query Test")
print("=" * 60)

query = input("\nEnter an ambiguous query: ").strip()

if not query:
    print("Please enter a query.")
    raise SystemExit

try:
    response = agent.invoke({
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
    })

    messages = response.get("messages", [])

    if not messages:
        print("No response generated.")
        raise SystemExit

    # Find tools selected by the agent
    tools_used = []

    for message in messages:
        tool_calls = getattr(message, "tool_calls", None)

        if not tool_calls:
            continue

        for tool_call in tool_calls:
            tool_name = tool_call.get("name")

            if tool_name and tool_name not in tools_used:
                tools_used.append(tool_name)

    final_answer = messages[-1].content

    print("\n" + "=" * 60)
    print("QUERY")
    print("=" * 60)
    print(query)

    print("\nTOOLS SELECTED")
    print("=" * 60)

    if tools_used:
        print(", ".join(tools_used))
    else:
        print("No tool selected")

    print("\nFINAL ANSWER")
    print("=" * 60)
    print(final_answer)

except Exception as e:
    print("\nTest Error:")
    print(str(e))