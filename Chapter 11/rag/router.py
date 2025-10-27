# rag/router.py

from langchain.schema import HumanMessage, SystemMessage
from rag.ollama_llm import get_llm
from rag.prompts import router_instructions
from rag.utils import safe_json_parse  # Utility to safely parse JSON from LLM

llm_json_mode = get_llm()

def route_question_and_get_source(question: str) -> str:
    """
    Routes question to either web or vectorstore based on LLM interpretation.
    Returns either "web" or "pdf".
    """
    messages = [
        SystemMessage(content=router_instructions),
        HumanMessage(content=question)
    ]

    response = llm_json_mode.invoke(messages)

    try:
        result = safe_json_parse(response.content)
        datasource = result.get("datasource", "vectorstore").lower()
        print(f"[ROUTER] Decision: {datasource}")
        return "web" if datasource == "websearch" else "pdf"
    except Exception as e:
        print(f"[ROUTER] JSON parsing failed: {e}")
        return "pdf"  # Fallback to document-based
