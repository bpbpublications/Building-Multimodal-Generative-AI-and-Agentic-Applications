from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import HumanMessage, ToolMessage

from tools.generate_cypher import generate_cypher
from tools.query_neo4j import run_cypher_and_get_results
from models.summarize_response import summarize_recommendations

from pydantic import BaseModel
from typing import Any

# === Define State Schema ===
class AgentState(BaseModel):
    question: str
    schema: str
    cypher_query: str = ""
    query_results: Any = None
    final_answer: str = ""

# Tool 1: Text2Cypher via Ollama

def cypher_tool_node(state: AgentState) -> dict:
    cypher = generate_cypher(state.question, state.schema)
    return {
        "cypher_query": cypher,
        "question": state.question,
        "schema": state.schema
    }

# Tool 2: Run Cypher on Neo4j

def neo4j_tool_node(state: AgentState) -> dict:
    results = run_cypher_and_get_results(state.cypher_query, top_k=5)
    return {
        "query_results": results,
        "cypher_query": state.cypher_query
    }

# Tool 3: Summarize Results via Ollama Mistral

def summarize_tool_node(state: AgentState) -> dict:
    summary = summarize_recommendations(state.query_results)
    return {"final_answer": summary}

# === LangGraph stateful agent ===
graph = StateGraph(AgentState)

# Define flow
graph.add_node("generate_cypher", RunnableLambda(cypher_tool_node))
graph.add_node("query_neo4j", RunnableLambda(neo4j_tool_node))
graph.add_node("summarize", RunnableLambda(summarize_tool_node))

# Set edges
graph.set_entry_point("generate_cypher")
graph.add_edge("generate_cypher", "query_neo4j")
graph.add_edge("query_neo4j", "summarize")
graph.add_edge("summarize", END)

# Compile app
app = graph.compile()

# === Example Run ===
if __name__ == "__main__":
    user_question = "Suggest a romantic movie featuring Mina Patel"
    schema = """
    (:Movie {title, genre, mood, release_year})
    (:Actor {name})-[:ACTED_IN]->(:Movie)
    (:Director {name})-[:DIRECTED]->(:Movie)
    (:Platform {name})<-[:AVAILABLE_ON]-(:Movie)
    """

    inputs = {
        "question": user_question,
        "schema": schema
    }

    result = app.invoke(inputs)

    print("\n✅ Final Answer:")
    print(result['final_answer'])
