
from langchain.embeddings import OllamaEmbeddings
from config import EMBEDDING_MODEL

def get_embedding_model():
    return OllamaEmbeddings(model=EMBEDDING_MODEL)
