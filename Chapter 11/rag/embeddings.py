from langchain_nomic.embeddings import NomicEmbeddings

def get_embeddings():
    return NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local")
