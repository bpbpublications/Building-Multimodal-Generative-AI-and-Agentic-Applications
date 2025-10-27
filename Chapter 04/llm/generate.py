from langchain.chat_models import ChatOpenAI
from config import MODEL_NAME, OPENAI_API_KEY

def get_llm():
    return ChatOpenAI(
        model=MODEL_NAME,
        temperature=0.2,
        api_key=OPENAI_API_KEY
    )