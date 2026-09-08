from fastapi import (
    FastAPI,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
)

import json

from agent import (
    run_agent,
    stream_agent_answer,
    PROMPT_VERSION,
)

from schemas import (
    AgentQueryRequest,
    AgentQueryResponse,
    HistoryItem,
)


# ============================================================
# 1. FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Agentic RAG System",
    description=(
        "Agentic RAG system using Hybrid Retrieval, "
        "Wikipedia, DuckDuckGo and Base/Fine-tuned Qwen model"
    ),
    version="2.3.0",
)


# ============================================================
# 2. IN-MEMORY HISTORY
# ============================================================

history: list[HistoryItem] = []


# ============================================================
# 3. HOME
# ============================================================

@app.get("/")
def home():

    return {
        "message": "Agentic RAG API is running",

        "models": {
            "base": "Qwen/Qwen2.5-1.5B-Instruct",
            "finetuned": (
                "Qwen/Qwen2.5-1.5B-Instruct + QLoRA"
            ),
        },

        "tools": [
            "search_documents",
            "search_wikipedia",
            "search_duckduckgo",
        ],

        "websocket": "/ws/agent",

        "prompt_version": PROMPT_VERSION,
    }


# ============================================================
# 4. NORMAL HTTP QUERY
# ============================================================

@app.post(
    "/agent/query",
    response_model=AgentQueryResponse,
)
def query_agent(
    request: AgentQueryRequest,
):

    try:

        result = run_agent(
            request.query
        )

        answer = result.get(
            "answer",
            "",
        )

        tools_used = result.get(
            "tools_used",
            [],
        )

        sources = result.get(
            "sources",
            [],
        )

        if not answer:

            answer = (
                "The agent could not generate "
                "an answer."
            )

        if not isinstance(
            answer,
            str,
        ):

            answer = str(
                answer
            )

        # ----------------------------------------------------
        # Store history
        # ----------------------------------------------------

        history.append(
            HistoryItem(
                query=request.query,
                answer=answer,
                tools_used=tools_used,
                sources=sources,
            )
        )

        if len(history) > 10:

            del history[:-10]

        return AgentQueryResponse(
            answer=answer,
            tools_used=tools_used,
            sources=sources,
        )

    except Exception as e:

        print(
            f"Agent Error: {e}"
        )

        raise HTTPException(
            status_code=500,
            detail=f"Agent failed: {str(e)}",
        )


# ============================================================
# 5. WEBSOCKET STREAMING
# ============================================================

@app.websocket(
    "/ws/agent"
)
async def websocket_agent(
    websocket: WebSocket,
):

    await websocket.accept()

    print(
        "WebSocket client connected."
    )

    try:

        while True:

            # =================================================
            # RECEIVE REQUEST
            # =================================================

            raw_message = (
                await websocket.receive_text()
            )

            raw_message = raw_message.strip()

            # -------------------------------------------------
            # Parse JSON from frontend
            # -------------------------------------------------

            try:

                request_data = json.loads(
                    raw_message
                )

                question = str(
                    request_data.get(
                        "query",
                        "",
                    )
                ).strip()

                model_type = request_data.get(
                    "model_type",
                    "finetuned",
                )

            except json.JSONDecodeError:

                # ------------------------------------------------
                # Backward compatibility:
                # If a plain text question is sent,
                # use fine-tuned model.
                # ------------------------------------------------

                question = raw_message

                model_type = "finetuned"

            # =================================================
            # VALIDATE MODEL
            # =================================================

            if model_type not in [
                "base",
                "finetuned",
            ]:

                await websocket.send_json(
                    {
                        "type": "error",
                        "message": (
                            "Invalid model_type. "
                            "Use 'base' or 'finetuned'."
                        ),
                    }
                )

                continue

            # =================================================
            # MODEL NAME
            # =================================================

            if model_type == "base":

                model_name = (
                    "Base Qwen "
                    "(Qwen2.5-1.5B-Instruct)"
                )

            else:

                model_name = (
                    "Fine-tuned Qwen "
                    "(QLoRA)"
                )

            # =================================================
            # LOG REQUEST
            # =================================================

            print(
                f"WebSocket question: {question}"
            )

            print(
                f"Selected model: {model_type}"
            )

            # =================================================
            # EXIT
            # =================================================

            if question.lower() == "exit":

                await websocket.send_json(
                    {
                        "type": "close",
                        "message": "Goodbye!",
                    }
                )

                await websocket.close()

                break

            # =================================================
            # EMPTY QUESTION
            # =================================================

            if not question:

                await websocket.send_json(
                    {
                        "type": "error",
                        "message": (
                            "Question cannot be empty."
                        ),
                    }
                )

                continue

            # =================================================
            # START
            # =================================================

            await websocket.send_json(
                {
                    "type": "start",
                    "question": question,
                    "model_type": model_type,
                    "model_name": model_name,
                    "prompt_version": PROMPT_VERSION,
                }
            )

            print(
                "WebSocket processing started."
            )

            # =================================================
            # STREAM AGENT
            # =================================================

            final_answer = ""

            tools_used = []

            agent_error = False

            sources = []

            try:

                # ------------------------------------------------
                # IMPORTANT:
                # Pass selected model to agent
                # ------------------------------------------------

                for event in stream_agent_answer(
                    question,
                    model_type=model_type,
                ):

                    if not event:

                        continue

                    event_type = event.get(
                        "type"
                    )

                    # ---------------------------------------------
                    # TOKEN
                    # ---------------------------------------------

                    if event_type == "token":

                        token = event.get(
                            "content",
                            "",
                        )

                        if token:

                            final_answer += token

                            await websocket.send_json(
                                {
                                    "type": "token",
                                    "content": token,
                                }
                            )

                    # ---------------------------------------------
                    # METADATA
                    # ---------------------------------------------

                    elif event_type == "metadata":

                        tools_used = event.get(
                            "tools_used",
                            [],
                        )

                        sources = event.get(
                            "sources",
                            [],
                        )

                        prompt_version = event.get(
                            "prompt_version",
                            PROMPT_VERSION,
                        )

                        await websocket.send_json(
                            {
                                "type": "metadata",

                                "tools_used": (
                                    tools_used
                                ),

                                "sources": (
                                    sources
                                ),

                                "prompt_version": (
                                    prompt_version
                                ),

                                "model_type": (
                                    model_type
                                ),

                                "model_name": (
                                    model_name
                                ),
                            }
                        )

                    # ---------------------------------------------
                    # ERROR FROM AGENT
                    # ---------------------------------------------

                    elif event_type == "error":

                        error_message = event.get(
                            "message",
                            "Agent streaming failed.",
                        )

                        print(
                            f"Streaming error: "
                            f"{error_message}"
                        )

                        await websocket.send_json(
                            {
                                "type": "error",
                                "message": error_message,
                            }
                        )

                        # Stop processing only this question.
                        # The outer WebSocket loop remains alive.
                        agent_error = True
                        break

                # If the agent failed, wait for the next question
                # instead of sending a misleading "done" event.
                if agent_error:
                    continue

                # =================================================
                # DONE
                # =================================================

                await websocket.send_json(
                    {
                        "type": "done",

                        "model_type": model_type,

                        "model_name": model_name,
                    }
                )

                print(
                    "WebSocket response completed."
                )

                # =================================================
                # SAVE HISTORY
                # =================================================

                if final_answer:

                    history.append(
                        HistoryItem(
                            query=question,
                            answer=final_answer,
                            tools_used=tools_used,
                            sources=sources,
                        )
                    )

                    if len(history) > 10:

                        del history[:-10]

            except Exception as e:

                print(
                    f"Streaming Agent Error: {e}"
                )

                try:

                    await websocket.send_json(
                        {
                            "type": "error",
                            "message": str(e),
                        }
                    )

                except Exception:

                    pass

    except WebSocketDisconnect:

        print(
            "WebSocket client disconnected."
        )

    except Exception as e:

        print(
            f"WebSocket Error: {e}"
        )

        try:

            await websocket.send_json(
                {
                    "type": "error",
                    "message": str(e),
                }
            )

        except Exception:

            pass


# ============================================================
# 6. HISTORY
# ============================================================

@app.get(
    "/agent/history",
    response_model=list[HistoryItem],
)
def get_history():

    return list(
        reversed(history)
    )