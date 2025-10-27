# tasks/summarizer.py

from core.embeddings import embed_text
from core.chroma_index import row_data_collection
from tasks.utils import ollama_completion
import datetime
import re
from typing import Any
from pydantic import BaseModel, field_validator

class SummarizeInput(BaseModel):
    user_data: Any
    db_name: str = None
    table_name: str = None

    @field_validator("user_data")
    @classmethod
    def validate_user_data(cls, v):
        if not isinstance(v, list):
            raise ValueError("user_data must be a list of rows")
        return v

def pre_filter(query: str):
    """
    Vector search for row summaries. Accepts a natural language query string.
    Supports basic age filtering like: 'users between 30 and 40'.
    """
    # Try to extract age range
    match = re.search(r"age[s]? (between|from)?\s*(\d+)\s*(and|to)?\s*(\d+)", query.lower())
    age_filter = None
    if match:
        age_filter = (int(match.group(2)), int(match.group(4)))

    query_embedding = embed_text(query)
    results = row_data_collection.query(query_embeddings=[query_embedding.tolist()], n_results=20)

    filtered = []
    for item in results["metadatas"][0]:
        summary = item.get("summary", "").lower()
        if age_filter:
            if any(age_filter[0] <= int(s) <= age_filter[1] for s in re.findall(r'\b\d{2}\b', summary)):
                filtered.append(item)
        else:
            filtered.append(item)

    return filtered

def summarize_and_store(input: SummarizeInput):
    if not input.user_data:
        return None

    summaries = []
    chunks = [input.user_data[i:i + 5] for i in range(0, len(input.user_data), 5)]

    for chunk in chunks:
        summary_text = str(chunk)
        response = ollama_completion(f"Summarize:\n{summary_text}", model="llama3")
        embedding = embed_text(response)
        row_ids = [str(row[0]) for row in chunk]
        row_data_collection.add(
            ids=[str(datetime.datetime.now().timestamp())],
            embeddings=[embedding.tolist()],
            metadatas=[{
                "summary": response,
                "original_data": str(chunk),
                "row_ids": row_ids,
                "db": input.db_name,
                "table": input.table_name
            }]
        )
        summaries.append(response)

    return summaries

