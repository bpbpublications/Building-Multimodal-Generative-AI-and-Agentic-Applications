import mlflow
from mlflow.entities import SpanType
from mlflow.tracing.utils import set_span_chat_messages
from functools import wraps
import ollama

# === Patch ollama.chat to add MLflow tracing ===
def trace_ollama_chat(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        with mlflow.start_span(name="ollama.chat", span_type=SpanType.CHAT_MODEL) as span:
            messages = kwargs.get("messages", [])
            model = kwargs.get("model", "mistral")
            span.set_inputs({"messages": messages, "model": model})
            set_span_chat_messages(span, messages)

            response = func(*args, **kwargs)

            span.set_outputs(response)
            if "message" in response:
                set_span_chat_messages(
                    span, [{"role": "assistant", "content": response["message"]["content"]}], append=True
                )
            return response
    return wrapper

# Patch the ollama.chat function
ollama.chat = trace_ollama_chat(ollama.chat)
