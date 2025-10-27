# streamlit_app/app.py

import streamlit as st
from retriever.reranker_and_llm import rerank_and_generate

st.set_page_config(page_title="Multimodal RAG Query Assistant", layout="wide")
st.title("Multimodal RAG Query Assistant")

# --- Optional: Future extension for image upload
uploaded_image = st.file_uploader("Upload an image (optional, not used yet)", type=["jpg", "png"])

# --- User text input
query = st.text_input("Ask a question about your documents and images:",
                      placeholder="e.g., What laptops support AI workloads?")

if st.button("Submit") and query:
    with st.spinner("Querying and reranking..."):
        try:
            response, reranked_results = rerank_and_generate(query_text=query)

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