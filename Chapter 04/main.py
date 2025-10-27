from orchestrator.rag_chain import get_rag_chain

print("RAG System Ready. Type 'exit' to quit.")
invoke_rag_chain = get_rag_chain()

while True:
    query = input("\nUser: ")
    if query.lower() in ['exit', 'quit']:
        break

    result = invoke_rag_chain(query)

    print("\nAssistant:", result["answer"])
    print("\nSources:")
    for doc in result.get("source_documents", []):
        print("-", doc.metadata.get("source", "[unknown]"))