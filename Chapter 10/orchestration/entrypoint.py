from retriever.reranker_and_llm import rerank_and_generate
from embedding.embedding_pipeline import refresh_embeddings, create_collection
from utils.data_loader import load_texts, load_images


import os

def initialize():
    print("\n[+] Creating Qdrant collection (if not exists)...")
    create_collection()
    print("[+] Refreshing vector database with latest image-text embeddings...")
    refresh_embeddings("data/images", "data/documents")
    print("[+] Initialization complete. Ready for queries.\n")

def interactive_loop():
    print("Multimodal RAG System (ReAct + Mistral + Qdrant)")
    print("Type 'exit' to quit.\n")
    while True:
        query = input("> Enter your query: ")
        if query.lower() in ("exit", "quit"):
            break
        try:
            response = rerank_and_generate(query_text=query)
            print("\n[Response]\n" + response + "\n")
        except Exception as e:
            print(f"[Error] {e}\n")

if __name__ == "__main__":
    initialize()
    interactive_loop()
