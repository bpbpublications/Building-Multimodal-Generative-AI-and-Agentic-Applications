import mlflow
import json
import traceback
import mlflow_ollama_patch  # Enables ollama.chat() tracing
from agent.langgraph_agent import app
from utils.ollama_metrics import evaluate_faithfulness_with_ollama, evaluate_relevance_with_ollama

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

    result = {}
    cypher_query = ""
    query_results = []
    final_answer = ""

    with mlflow.start_run(run_name="CypherTest_Run1") as run:
        print("✅ Started MLflow run...")
        print("Run ID:", run.info.run_id)

        try:
            result = app.invoke(inputs)
            cypher_query = result.get("cypher_query", "")
            query_results = result.get("query_results", [])
            final_answer = result.get("final_answer", "")

            # === Print Outputs ===
            print("\n📌 1. Cleaned Cypher Query:")
            print(cypher_query)
            print("\n🧠 2. Raw Results from Graph DB:")
            print(json.dumps(query_results, indent=2))
            print("\n🗣️ 3. Final Answer (Mistral):")
            print(final_answer)

            # === Log to MLflow ===
            mlflow.log_param("question", user_question)
            mlflow.log_param("cypher_query", cypher_query or "EMPTY")
            mlflow.log_text(json.dumps(query_results, indent=2), "neo4j_context.json")
            mlflow.log_text(final_answer or "EMPTY", "final_answer.txt")

            # === Evaluate ===
            faith_score = 0.0
            rel_score = 0.0

            try:
                faith_score = evaluate_faithfulness_with_ollama(user_question, json.dumps(query_results), final_answer)
            except Exception as eval_e:
                print("⚠️ Faithfulness evaluation failed:", eval_e)
                faith_score = 0.0

            try:
                rel_score = evaluate_relevance_with_ollama(user_question, final_answer)
            except Exception as eval_e:
                print("⚠️ Relevance evaluation failed:", eval_e)
                rel_score = 0.0

            mlflow.log_metric("faithfulness", faith_score)
            mlflow.log_metric("relevance", rel_score)

        except Exception as e:
            print("❌ Exception during run:")
            traceback.print_exc()
            mlflow.log_param("run_failed", "1")
            mlflow.log_metric("faithfulness", 0.0)
            mlflow.log_metric("relevance", 0.0)

        finally:
            print("✅ Ending MLflow run...")
            mlflow.end_run()
