import os
from typing import TypedDict

import chromadb
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from langgraph.graph import StateGraph, END


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")

COLLECTION_NAME = "zepto_policies"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

MOCK_LLM = os.getenv("MOCK_LLM", "1") == "1"


# ---------------------------------------------------------
# Load embedding model and ChromaDB
# ---------------------------------------------------------

print("Loading embedding model...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

print("Opening ChromaDB...")
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)

collection = chroma_client.get_collection(
    name=COLLECTION_NAME
)


# ---------------------------------------------------------
# State
# ---------------------------------------------------------

class AssistantState(TypedDict, total=False):
    query: str
    intent: str
    answer: str
    sources: list
    confidence: float


# ---------------------------------------------------------
# Final response schema
# ---------------------------------------------------------

class AssistantResponse(BaseModel):
    answer: str
    sources: list = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


# ---------------------------------------------------------
# Node 1: classify_intent
# ---------------------------------------------------------

def classify_intent(state: AssistantState) -> AssistantState:
    query = state["query"].lower()

    policy_keywords = [
        "delivery",
        "return",
        "refund",
        "membership",
        "tracking",
        "cancel",
        "gift card",
        "support hours",
    ]

    if any(keyword in query for keyword in policy_keywords):
        intent = "policy_question"
    else:
        intent = "general_question"

    return {
        **state,
        "intent": intent,
    }

# ---------------------------------------------------------
# Prompt template
# ---------------------------------------------------------

PROMPT_TEMPLATE = """
Role:
You are a Zepto customer support assistant.

Context:
Use only the retrieved Zepto policy context provided below.

Task:
Answer the user's question using the retrieved context.

Format:
Give a concise, direct answer.

Length:
Keep the response within 2-4 sentences.

Negative constraint:
Do not invent, assume, or provide Zepto policies that are not present
in the retrieved context.

Few-shot example:
User: What is the delivery time?
Context: Zepto delivers grocery and household essentials to serviceable
pin codes within 10 to 30 minutes.
Answer: Zepto typically delivers within 10 to 30 minutes to serviceable
pin codes.

User question:
{query}

Retrieved context:
{context}
"""

# ---------------------------------------------------------
# Node 2: retrieve_and_answer
# ---------------------------------------------------------

def retrieve_and_answer(state: AssistantState) -> AssistantState:
    query = state["query"]

    # Always embed the query and retrieve top 3 results.
    query_embedding = embedding_model.encode(
        query,
        normalize_embeddings=True,
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
    )

    retrieved_documents = results.get("documents", [[]])[0]
    retrieved_metadatas = results.get("metadatas", [[]])[0]

    if retrieved_documents:
        top_chunk = retrieved_documents[0]

        # Keep the mock response deterministic.
        snippet = top_chunk[:500]

        answer = f"Based on the retrieved context: {snippet}"

        sources = [
            metadata.get("source", "")
            for metadata in retrieved_metadatas
            if metadata
        ]

        confidence = 0.9

    else:
        answer = "I could not find relevant information in the policy documents."
        sources = []
        confidence = 0.2

    return {
        **state,
        "answer": answer,
        "sources": sources,
        "confidence": confidence,
    }


# ---------------------------------------------------------
# Node 3: direct_answer
# ---------------------------------------------------------

def direct_answer(state: AssistantState) -> AssistantState:
    answer = "I can only answer questions about Zepto policies right now."

    return {
        **state,
        "answer": answer,
        "sources": [],
        "confidence": 0.8,
    }


# ---------------------------------------------------------
# Conditional routing
# ---------------------------------------------------------

def route_after_classification(state: AssistantState):
    if state["intent"] == "policy_question":
        return "retrieve_and_answer"

    return "direct_answer"


# ---------------------------------------------------------
# Build LangGraph
# ---------------------------------------------------------

builder = StateGraph(AssistantState)

builder.add_node(
    "classify_intent",
    classify_intent,
)

builder.add_node(
    "retrieve_and_answer",
    retrieve_and_answer,
)

builder.add_node(
    "direct_answer",
    direct_answer,
)

builder.set_entry_point("classify_intent")

builder.add_conditional_edges(
    "classify_intent",
    route_after_classification,
    {
        "retrieve_and_answer": "retrieve_and_answer",
        "direct_answer": "direct_answer",
    },
)

builder.add_edge(
    "retrieve_and_answer",
    END,
)

builder.add_edge(
    "direct_answer",
    END,
)

graph = builder.compile()


# ---------------------------------------------------------
# Public function
# ---------------------------------------------------------

def ask_assistant(query: str) -> AssistantResponse:
    result = graph.invoke(
        {
            "query": query,
        }
    )

    return AssistantResponse(
        answer=result["answer"],
        sources=result.get("sources", []),
        confidence=result.get("confidence", 0.0),
    )