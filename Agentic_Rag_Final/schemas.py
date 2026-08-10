from pydantic import BaseModel, Field


class AgentQueryRequest(BaseModel):
    query: str = Field(..., min_length=1)


class AgentQueryResponse(BaseModel):
    answer: str
    tools_used: list[str]
    sources: list[str]


class HistoryItem(BaseModel):
    query: str
    answer: str
    tools_used: list[str]
    sources: list[str]