
from langchain_community.vectorstores import Chroma
from embeddings.embedder import get_embedding_model
from config import VECTOR_DB_PATH

def get_vectorstore(documents):
    embedding_model = get_embedding_model()
    return Chroma.from_documents(documents, embedding=embedding_model, persist_directory=VECTOR_DB_PATH)
