from langchain.retrievers import BM25Retriever, EnsembleRetriever

def get_hybrid_retriever(chunks, vectorstore):
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 4
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    return EnsembleRetriever(retrievers=[bm25_retriever, vector_retriever], weights=[0.5, 0.5])