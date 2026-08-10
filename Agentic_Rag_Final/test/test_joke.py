from tools.joke_tool import get_joke

print("=" * 60)
print("Joke API Test")
print("=" * 60)

input("\nPress Enter to get a joke...")

try:
    result = get_joke()

    print("\nJoke:")
    print("-" * 60)
    print(result)

except Exception as e:
    print(f"\nJoke API Error: {e}")