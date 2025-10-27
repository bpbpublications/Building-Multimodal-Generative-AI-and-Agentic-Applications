
import sys
import os

# Adjust path to ensure `orchestrator` and others are found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from orchestrator.rag_chain import get_rag_chain

print("RAG System Ready. Type 'exit' to quit.")
rag_chain = get_rag_chain()

while True:
    query = input("\nUser: ")
    if query.lower() in ['exit', 'quit']:
        break

    result = rag_chain.invoke({"question": query})
    print("\nAssistant:", result["answer"])