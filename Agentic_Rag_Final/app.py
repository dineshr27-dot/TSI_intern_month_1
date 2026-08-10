import requests
import streamlit as st


# ============================================================
# Configuration
# ============================================================

API_URL = "http://127.0.0.1:8000"


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="Agentic RAG Assistant",
    page_icon="🤖",
    layout="centered"
)


# ============================================================
# Title
# ============================================================

st.title("🤖 Agentic RAG Assistant")

st.caption(
    "Hybrid RAG • Wikipedia • DuckDuckGo • Joke API"
)


# ============================================================
# Session State
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# Display Previous Messages
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        # Display tools for assistant messages
        if message["role"] == "assistant":

            tools = message.get("tools_used", [])

            if tools:
                st.caption(
                    "🛠️ Tools used: "
                    + ", ".join(tools)
                )

            sources = message.get("sources", [])

            if sources:

                with st.expander("📚 Retrieved Documents"):

                    for source in sources:
                        st.write(source)


# ============================================================
# User Input
# ============================================================

query = st.chat_input(
    "Ask something..."
)


if query:

    # --------------------------------------------------------
    # Store User Message
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": query
        }
    )

    # --------------------------------------------------------
    # Display User Message
    # --------------------------------------------------------

    with st.chat_message("user"):
        st.markdown(query)

    # --------------------------------------------------------
    # Call FastAPI
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            try:

                response = requests.post(
                    f"{API_URL}/agent/query",
                    json={
                        "query": query
                    },
                    timeout=120
                )

                # --------------------------------------------
                # Successful Request
                # --------------------------------------------

                if response.status_code == 200:

                    data = response.json()

                    answer = data.get(
                        "answer",
                        "No answer received."
                    )

                    tools_used = data.get(
                        "tools_used",
                        []
                    )

                    sources = data.get(
                        "sources",
                        []
                    )

                    # Display Answer
                    st.markdown(answer)

                    # Display Tools
                    if tools_used:

                        st.caption(
                            "🛠️ Tools used: "
                            + ", ".join(tools_used)
                        )

                    # Display Sources
                    if sources:

                        with st.expander(
                            "📚 Sources"
                        ):

                            for source in sources:
                                st.write(source)

                    # Store Assistant Message
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "tools_used": tools_used,
                            "sources": sources
                        }
                    )

                # --------------------------------------------
                # API Error
                # --------------------------------------------

                else:

                    error_message = (
                        f"API Error: "
                        f"{response.status_code}"
                    )

                    try:
                        error_data = response.json()

                        error_message += (
                            f"\n\n"
                            f"{error_data.get('detail', '')}"
                        )

                    except ValueError:
                        pass

                    st.error(error_message)

            # ------------------------------------------------
            # FastAPI Not Running
            # ------------------------------------------------

            except requests.exceptions.ConnectionError:

                st.error(
                    "Cannot connect to the FastAPI server. "
                    "Make sure FastAPI is running."
                )

            # ------------------------------------------------
            # Timeout
            # ------------------------------------------------

            except requests.exceptions.Timeout:

                st.error(
                    "The request took too long. "
                    "Please try again."
                )

            # ------------------------------------------------
            # Other Errors
            # ------------------------------------------------

            except Exception as e:

                st.error(
                    f"Something went wrong: {str(e)}"
                )


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:

    st.header("Agentic RAG")

    st.write(
        "The agent automatically selects the "
        "appropriate tool for your question."
    )

    st.write("**Available Tools**")

    st.write("📄 Hybrid Document Search")
    st.write("🌐 Wikipedia")
    st.write("🔎 DuckDuckGo")
    st.write("😂 Joke API")

    st.divider()

    # --------------------------------------------------------
    # Clear Chat
    # --------------------------------------------------------

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()