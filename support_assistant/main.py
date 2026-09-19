from pydantic import BaseModel
from fastapi import FastAPI

from graph import AssistantResponse, ask_assistant


app = FastAPI(
    title="Zepto Support Assistant",
    description="GenAI support assistant using LangGraph, ChromaDB, and local embeddings.",
    version="1.0.0",
)


class AskRequest(BaseModel):
    query: str


@app.get("/")
def root():
    return {
        "message": "Zepto Support Assistant is running"
    }


@app.post("/ask", response_model=AssistantResponse)
def ask(request: AskRequest):
    return ask_assistant(request.query)