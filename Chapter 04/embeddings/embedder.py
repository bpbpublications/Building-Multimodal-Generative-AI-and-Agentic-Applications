from langchain.embeddings import OpenAIEmbeddings
from config import EMBEDDING_MODEL, OPENAI_API_KEY

def get_embedding_model():
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=OPENAI_API_KEY
    )