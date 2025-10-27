# memory/conversation_buffer.py
from langchain.memory import ConversationBufferMemory

# Store only the LLM’s answer (not the source-document list)
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"      # ← tell the memory which output to keep
)
