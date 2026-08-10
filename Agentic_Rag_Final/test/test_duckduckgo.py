from tools.duckduckgo_tool import search_duckduckgo

print("=" * 60)
print("DuckDuckGo API Test")
print("=" * 60)

query = input("\nEnter web search query: ").strip()

if not query:
    print("Please enter a query.")
else:
    try:
        result = search_duckduckgo(query)

        print("\nDuckDuckGo Results:")
        print("-" * 60)
        print(result)

    except Exception as e:
        print(f"\nDuckDuckGo Error: {e}")