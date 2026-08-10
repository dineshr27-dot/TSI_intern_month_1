from ddgs import DDGS


def search_duckduckgo(query: str) -> str:
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))

        if not results:
            return "No results found."

        output = ""

        for i, result in enumerate(results, start=1):
            output += (
                f"{i}. {result['title']}\n"
                f"{result['body']}\n"
                f"{result['href']}\n\n"
            )

        return output

    except Exception as e:
        return str(e)