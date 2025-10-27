from config import SUMMARIZER_MODEL
from langchain_ollama import OllamaLLM

def summarize_recommendations(raw_results: list) -> str:
    formatted = "\n".join([str(r) for r in raw_results])
    prompt = f"""You are a summarization assistant.
Only use the following results extracted from a knowledge graph. 
**Do not invent or infer any new movies or information.** 
Your job is to rephrase and present the recommendations clearly for the user.

### Knowledge Graph Results:
{formatted}

### Summary:
"""
    model = OllamaLLM(model=SUMMARIZER_MODEL)
    return model.invoke(prompt).strip()
