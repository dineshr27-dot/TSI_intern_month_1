from tools.wikipedia_tool import search_wikipedia

print("=" * 60)
print("Wikipedia API Test")
print("=" * 60)

query = input("\nEnter Wikipedia query: ").strip()

if not query:
    print("Please enter a query.")
else:
    try:
        result = search_wikipedia(query)

        print("\nWikipedia Result:")
        print("-" * 60)
        print(result)

    except Exception as e:
        print(f"\nWikipedia Error: {e}")