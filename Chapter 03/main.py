from orchestrator.rag_chain import get_rag_chain
from memory.conversation_buffer import memory

print("RAG System Ready. Type 'exit' to quit.")
rag_chain = get_rag_chain()

while True:
    query = input("\nUser: ")
    if query.lower() in ['exit', 'quit']:
        break

    # Use invoke() instead of deprecated __call__
    result = rag_chain.invoke({"question": query})

    print("\nAssistant:", result["answer"])

    print("\nSources:")
    for doc in result.get("source_documents", []):
        print("-", doc.metadata.get("source", "[unknown]"))
