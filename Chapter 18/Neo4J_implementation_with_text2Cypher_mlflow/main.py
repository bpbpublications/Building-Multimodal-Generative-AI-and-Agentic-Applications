import mlflow
import json
from agent.langgraph_agent import app
from mlflow.metrics.genai import faithfulness, relevance



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

    # === Call LangGraph Agent ===
    result = app.invoke(inputs)

    cypher_query = result["cypher_query"]
    query_results = result["query_results"]
    final_answer = result["final_answer"]

    # === Print Outputs ===
    print("\n📌 1. Cleaned Cypher Query:")
    print(cypher_query)

    print("\n🧠 2. Raw Results from Graph DB:")
    print(json.dumps(query_results, indent=2))

    print("\n🗣️ 3. Final Answer (Mistral):")
    print(final_answer)

    # === MLflow Logging and Evaluation ===
    with mlflow.start_run(run_name="CypherTest_Run_2") as run:
        print("\n📅 Started MLflow run:", run.info.run_id)

        # Log params & artifacts
        mlflow.log_param("question", user_question)
        mlflow.log_param("cypher_query", cypher_query or "EMPTY")
        mlflow.log_text(json.dumps(query_results, indent=2), "neo4j_context.json")
        mlflow.log_text(final_answer or "EMPTY", "final_answer.txt")

        # === Evaluate using Ollama as the judge ===
        try:
            faith = faithfulness(model="ollama:/mistral")( 
                predictions=[final_answer],
                inputs=[user_question],
                context=[json.dumps(query_results)]
            ).scores[0]
        except Exception as e:
            print("⚠️  Faithfulness evaluation failed:", e)
            faith = 0.0

        try:
            relevance_score = relevance(model="ollama:/mistral")( 
                predictions=[final_answer],
                inputs=[user_question]
            ).scores[0]
        except Exception as e:
            print("⚠️  Relevance evaluation failed:", e)
            relevance_score = 0.0

        mlflow.log_metric("faithfulness", faith)
        mlflow.log_metric("relevance", relevance_score)

        print("\n🚀 Metrics logged:", {
            "faithfulness": faith,
            "relevance": relevance_score
        })
