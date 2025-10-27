from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_community.vectorstores import Chroma

def filter_chunks_by_topic(chunks, topic):
    topic = topic.lower()
    if "blockchain" in topic or "crypto" in topic:
        return [c for c in chunks if "blockchain" in c.metadata.get("source", "").lower()]
    elif "education" in topic or "ai" in topic or "artificial intelligence" in topic:
        return [c for c in chunks if "education" in c.metadata.get("source", "").lower()]
    else:
        return chunks  # fallback: use all chunks

def get_hybrid_retriever(chunks, vectorstore, topic=None):
    # Filter chunks based on topic
    filtered_chunks = filter_chunks_by_topic(chunks, topic)

    bm25_retriever = BM25Retriever.from_documents(filtered_chunks)
    bm25_retriever.k = 4

    # Vector retriever with filtered docs — dynamic filtering not supported, so fallback on filtering full store
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    return EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5]
    )
