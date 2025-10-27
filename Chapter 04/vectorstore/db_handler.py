# vectorstore/db_handler.py
import os
from pathlib import Path
from langchain_community.vectorstores import FAISS
from embeddings.embedder import get_embedding_model
from config import VECTOR_DB_PATH

def get_vectorstore(documents):
    embedding_model = get_embedding_model()

    index_file = Path(VECTOR_DB_PATH) / "index.faiss"
    store_file = Path(VECTOR_DB_PATH) / "index.pkl"

    # Load only if both index and metadata exist
    if index_file.exists() and store_file.exists():
        return FAISS.load_local(
            VECTOR_DB_PATH,
            embedding_model,
            allow_dangerous_deserialization=True
        )

    # Otherwise, build and save a new index
    vectorstore = FAISS.from_documents(
        documents,
        embedding=embedding_model
    )
    vectorstore.save_local(VECTOR_DB_PATH)
    return vectorstore
