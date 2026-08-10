import requests


def get_joke() -> str:
    """Fetch one random safe joke from JokeAPI."""

    url = "https://v2.jokeapi.dev/joke/Any?safe-mode"

    try:
        response = requests.get(
            url,
            timeout=10
        )

        response.raise_for_status()

        data = response.json()

        if data.get("error"):
            return "Joke API returned an error."

        if data.get("type") == "single":
            return data.get(
                "joke",
                "No joke found."
            )

        if data.get("type") == "twopart":
            setup = data.get("setup", "")
            delivery = data.get("delivery", "")

            return f"{setup}\n{delivery}"

        return "No joke found."

    except requests.exceptions.Timeout:
        return "Joke API request timed out."

    except requests.exceptions.RequestException as e:
        return f"Joke API request failed: {str(e)}"

    except Exception as e:
        return f"Joke tool error: {str(e)}"