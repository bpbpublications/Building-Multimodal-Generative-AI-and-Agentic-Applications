from core.embeddings import embed_text
from core.chroma_index import row_data_collection
from tasks.utils import ollama_completion

def aggregate_summarized_data(query):
    query_embedding = embed_text(query)
    results = row_data_collection.query(query_embeddings=[query_embedding.tolist()], n_results=5)
    summaries = results["metadatas"][0]
    combined_text = " ".join([item['summary'] for item in summaries])
    final_summary = ollama_completion(f"Aggregate these summaries into a comprehensive insight:\n{combined_text}")
    return {"final_summary": final_summary, "details": summaries}