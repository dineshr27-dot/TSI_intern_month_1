from tools.wikipedia_tool import search_wikipedia
from tools.duckduckgo_tool import search_duckduckgo
from tools.joke_tool import get_joke

print("=" * 60)
print("Wikipedia")
print("=" * 60)

print(search_wikipedia("LangChain"))
print(search_wikipedia("Python programming language"))
print(search_wikipedia("about TamilNadu"))
print(search_wikipedia("C joeshp vijay"))

print("\n")

print("=" * 60)
print("DuckDuckGo")
print("=" * 60)

print(search_duckduckgo("Latest AI News"))

print("\n")

print("=" * 60)
print("Joke")
print("=" * 60)

print(get_joke())