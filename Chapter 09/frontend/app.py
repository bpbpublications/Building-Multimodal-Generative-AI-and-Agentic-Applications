# app.py
import streamlit as st
import os
from rag.embedding_utils import embed_text_ollama, embed_image_ollama
from rag.reranker import rerank
from rag.config import *
import chromadb

client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

st.title("💻 Multimodal RAG Laptop Assistant")
mode = st.radio("Choose Mode", ["Image → Specs", "Image + Text → Specs", "Text → Image + Specs"])

if mode == "Image → Specs":
    uploaded = st.file_uploader("Upload laptop image", type=["jpg", "png"])
    if uploaded:
        path = "temp_image.jpg"
        with open(path, "wb") as f:
            f.write(uploaded.getbuffer())
        st.image(path, caption="Input Image")
        vec = embed_image_ollama(path)
        collection = client.get_collection(name=CHROMA_IMAGE_COLLECTION)
        results = collection.query(query_embeddings=[vec], n_results=5, include=["metadatas"])
        specs = rerank("laptop image", results["metadatas"][0])
        if specs:
            file_name = specs[0].get("file")
            if file_name:
                matched_img_path = os.path.join(IMAGE_FOLDER, file_name)
                if os.path.exists(matched_img_path):
                    st.image(matched_img_path, caption="Closest Match")
                spec_file = file_name.replace(".jpg", ".txt")
                spec_path = os.path.join(TEXT_FOLDER, spec_file)
                if os.path.exists(spec_path):
                    with open(spec_path, "r") as f:
                        content = f.read()
                    st.text_area("Laptop Specs", content, height=300)
                else:
                    st.warning("Spec file not found.")
            else:
                st.warning("No matching specs found.")
        else:
            st.warning("No matching results found.")

elif mode == "Image + Text → Specs":
    uploaded = st.file_uploader("Upload image", type=["jpg", "png"])
    query = st.text_input("Describe the laptop")
    if uploaded and query:
        path = f"temp_image.jpg"
        with open(path, "wb") as f:
            f.write(uploaded.getbuffer())
        st.image(path, caption="Uploaded Image")
        image_vec = embed_image_ollama(path)
        text_vec = embed_text_ollama(query)
        joint_vec = [(i + j) / 2 for i, j in zip(image_vec, text_vec)]
        collection = client.get_collection(name=CHROMA_TEXT_COLLECTION)
        results = collection.query(query_embeddings=[joint_vec], n_results=5, include=["metadatas"])
        specs = rerank(query, results["metadatas"][0])
        if specs:
            file_name = specs[0].get("file")
            if file_name:
                matched_img_path = os.path.join(IMAGE_FOLDER, file_name.replace(".txt", ".jpg"))
                if os.path.exists(matched_img_path):
                    st.image(matched_img_path, caption="Closest Match")
                file_path = os.path.join(TEXT_FOLDER, file_name)
                if os.path.exists(file_path):
                    with open(file_path, "r") as f:
                        content = f.read()
                    st.text_area("Laptop Specs", content, height=300)
                else:
                    st.warning("Spec file not found.")
            else:
                st.warning("No matching specs found.")
        else:
            st.warning("No matching results found.")

elif mode == "Text → Image + Specs":
    query = st.text_input("Type your query")
    if query:
        vec = embed_text_ollama(query)
        collection = client.get_collection(name=CHROMA_TEXT_COLLECTION)
        results = collection.query(query_embeddings=[vec], n_results=5, include=["metadatas"])
        specs = rerank(query, results["metadatas"][0])
        file_name = specs[0].get("file", "unknown.jpg")
        img_path = os.path.join(IMAGE_FOLDER, file_name.replace(".txt", ".jpg"))
        if os.path.exists(img_path):
            st.image(img_path, caption="Closest Match")
        file_name = specs[0].get("file")
        if file_name:
            file_path = os.path.join(TEXT_FOLDER, file_name)
            if os.path.exists(file_path):
                with open(file_path, "r") as f:
                    content = f.read()
                st.text_area("Laptop Specs", content, height=300)
            else:
                st.warning("Spec file not found.")
        else:
            st.warning("No matching specs found.")