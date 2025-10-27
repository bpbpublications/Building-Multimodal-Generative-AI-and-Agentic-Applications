from agent.langgraph_agent import app
import json

if __name__ == "__main__":
    user_question = input("Ask your question: ")

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

    print("\n📌 1. Cleaned Cypher Query:")
    print(result["cypher_query"])

    print("\n🧠 2. Raw Results from Graph DB:")
    print(json.dumps(result["query_results"], indent=2))

    print("\n🗣️ 3. Final Answer (Mistral):")
    print(result["final_answer"])
