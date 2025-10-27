# ott_recommender/scripts/run_agent.py

from app.query_agent import build_agent, hybrid_retriever
from app.reranker import Reranker
from langchain_community.llms import Ollama

def run_reranker(query, candidates):
    print("\n[🔍 RERANKER OUTPUT]")
    reranker = Reranker()
    ranked = reranker.rank(query, candidates)
    if ranked:
        print("Top reranked recommendation (raw):", ranked[0])
        return ranked[0]
    else:
        print("No reranked result found.")
        return None

def explain_with_llm(query, top_reranked):
    print("\n[🧠 LLM Explanation of Reranked Result]")
    if not top_reranked:
        print("No explanation because reranker returned no result.")
        return
    llm = Ollama(model="mistral")
    prompt = (
        f"The user asked: '{query}'\n"
        f"Out of all available content, the most relevant match is:\n"
        f"{top_reranked}\n"
        f"Explain to the user in natural language why this is the best match."
    )
    response = llm.invoke(prompt)
    print(response)

def run_agent_loop():
    agent = build_agent()
    while True:
        query = input("Enter your OTT content query: ")
        if query.lower() in ('exit', 'quit'):
            print("\n[👋 Session Ended]")
            break

        print("\n[🤖 AGENT OUTPUT]")
        agent_response = agent.run(query)
        print("\nRecommended by Agent:", agent_response)

        # Extract raw results from hybrid retriever
        raw_candidates = hybrid_retriever(query)

        # Rerank
        top_reranked = run_reranker(query, raw_candidates)

        # LLM explanation
        explain_with_llm(query, top_reranked)

if __name__ == "__main__":
    run_agent_loop()