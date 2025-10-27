import os
from core.sql_executor import execute_sql
from core.chroma_index import row_data_collection
from core.embeddings import embed_text
from tasks.utils import ollama_completion
import datetime

def populate_chroma():
    print("🔄 Checking ChromaDB for existing entries...")
    existing = row_data_collection.count()
    if existing > 0:
        print(f"✅ ChromaDB already populated with {existing} entries. Skipping.")
        return

    print("🔄 Executing SQL to fetch rows from both DBs...")
    rows = execute_sql("SELECT * FROM customers")
    if rows:
        print(f"✅ Retrieved {len(rows)} rows. Now summarizing and embedding into ChromaDB...")

        # Split into manageable chunks
        chunks = [rows[i:i + 5] for i in range(0, len(rows), 5)]
        for chunk in chunks:
            # Prepare a readable summary prompt
            readable_chunk = [
                f"DB: {db}, ID: {row[0]}, Name: {row[1]}, Age: {row[2]}, City: {row[3]}"
                for db, row in chunk
            ]
            summary_text = "\n".join(readable_chunk)

            # Generate a summary using Ollama
            response = ollama_completion(f"Summarize:\n{summary_text}", model="llama3")

            # Compute the embedding from the summary
            embedding = embed_text(response)

            # Prepare ChromaDB-compatible metadata
            row_data_collection.add(
                ids=[str(datetime.datetime.now().timestamp())],
                embeddings=[embedding.tolist()],
                metadatas=[{
                    "summary": response,
                    "original_data": summary_text,
                    "row_ids": ", ".join([str(row[0]) for db, row in chunk]),
                    "db_names": ", ".join(set([db for db, row in chunk])),
                    "table": "customers"
                }]
            )
        print("✅ Summarization complete and stored in ChromaDB.")
    else:
        print("⚠️ No rows returned from execute_sql().")

if __name__ == "__main__":
    populate_chroma()
