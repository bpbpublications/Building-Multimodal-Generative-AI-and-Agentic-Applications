from langchain_ollama import OllamaLLM  # after installing updated lib

def get_llm():
    #return OllamaLLM(model="llama3.2:3b-instruct-fp16")
    return OllamaLLM(model="llama3")