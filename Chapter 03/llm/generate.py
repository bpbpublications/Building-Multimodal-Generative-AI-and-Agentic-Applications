
from langchain_community.llms import Ollama
from config import MODEL_NAME

def get_llm():
    return Ollama(model=MODEL_NAME, temperature=0.2)
