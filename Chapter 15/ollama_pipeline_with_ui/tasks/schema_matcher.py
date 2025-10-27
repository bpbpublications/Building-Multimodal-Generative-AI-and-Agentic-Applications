from core.embeddings import embed_text
from core.chroma_index import global_index

def schema_matching(query):
    query_embedding = embed_text(query)
    results = global_index.query(query_embeddings=[query_embedding.tolist()], n_results=5)
    return results["metadatas"][0]


# tasks/grader.py

from tasks.utils import ollama_completion

def grade_sql(sql_query):
    return ollama_completion(f"Grade this SQL query for correctness, relevance, and efficiency:\n{sql_query}")

def grade_summary(summary_text):
    return ollama_completion(f"Grade this summary for accuracy, comprehensiveness, and clarity:\n{summary_text}")