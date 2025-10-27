# streamlit_app/app.py

import streamlit as st
from retriever.reranker_and_llm import rerank_and_generate
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
import tempfile

st.set_page_config(page_title="Multimodal RAG Query Assistant", layout="wide")
st.title("📚🔍 Multimodal RAG Query Assistant")

# Load CLIP model for image embedding
clip_model_name = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(clip_model_name)
clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

def embed_uploaded_image(image_file):
    image = Image.open(image_file).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)
    return features[0].cpu().tolist()

# --- Optional: Image Upload
uploaded_image = st.file_uploader("Upload an image (optional)", type=["jpg", "png"])

# --- User text input
query = st.text_input("Ask a question about your documents and images:",
                      placeholder="e.g., What laptops support AI workloads?")

if st.button("Submit") and query:
    with st.spinner("Querying and reranking..."):
        try:
            image_vector = None
            if uploaded_image:
                image_vector = embed_uploaded_image(uploaded_image)

            response, reranked_results = rerank_and_generate(query_text=query, image_vector=image_vector)

            st.markdown("### 🔎 Answer")
            st.success(response)

            st.markdown("### 📄 Top Reranked Documents")
            for i, res in enumerate(reranked_results):
                st.markdown(f"**Result {i+1}:**")
                st.code(
                    res.get("text") or res.get("description") or res.get("content") or "⚠️ No content available",
                    language="markdown"
                )

        except Exception as e:
            st.error(f"Error: {str(e)}")

st.markdown("---")
st.markdown("Built with Qdrant + LangChain + Mistral (Ollama)")
