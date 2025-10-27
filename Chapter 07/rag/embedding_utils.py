from llama_index.embeddings.huggingface import HuggingFaceEmbedding

_MODEL_ID = "openai/clip-vit-base-patch32"

def get_mm_embedder(device: str = "cpu"):
    return HuggingFaceEmbedding(model_name=_MODEL_ID, device=device, trust_remote_code=True)
