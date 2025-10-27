def format_sources(source_documents):
    return [doc.metadata.get("source", "[unknown]") for doc in source_documents]
