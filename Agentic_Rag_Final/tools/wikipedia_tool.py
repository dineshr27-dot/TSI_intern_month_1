import wikipedia

wikipedia.set_lang("en")

def search_wikipedia(query: str) -> str:
    try:
        results = wikipedia.search(query)

        if not results:
            return "No Wikipedia page found."

        page = wikipedia.page(results[0], auto_suggest=False)
        return wikipedia.summary(page.title, sentences=3)

    except wikipedia.DisambiguationError as e:
        return f"Multiple pages found: {', '.join(e.options[:5])}"

    except wikipedia.PageError:
        return "No Wikipedia page found."

    except Exception as e:
        return str(e)