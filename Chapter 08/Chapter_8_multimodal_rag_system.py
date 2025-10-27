### retriever.py or index_builder.py ###

from typing import List, Tuple
from langchain.schema import Document
from rag.embedding_utils import get_mm_embedder
from rag.loaders import load_pdfs_and_texts, load_images
from qdrant_client import QdrantClient, models
from pathlib import Path
from numpy.linalg import norm
import os

DB_PATH = "data/qdrant_mm"
TEXT_COLLECTION = "vdr_text"
IMAGE_COLLECTION = "vdr_images"


def normalize(vecs):
    return [v / norm(v) for v in vecs]


def build_vectorstores():
    text_docs: List[Document] = load_pdfs_and_texts("data/documents")
    image_docs: List[Document] = load_images("data/images")

    embedder = get_mm_embedder()

    text_vecs = normalize(embedder.get_text_embedding_batch([d.page_content for d in text_docs]))
    image_vecs = normalize(embedder.get_image_embedding_batch([d.page_content for d in image_docs]))

    client = QdrantClient(path=DB_PATH)

    if not client.collection_exists(TEXT_COLLECTION):
        dim = len(text_vecs[0])
        client.create_collection(TEXT_COLLECTION, vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE))
    if not client.collection_exists(IMAGE_COLLECTION):
        dim = len(image_vecs[0])
        client.create_collection(IMAGE_COLLECTION, vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE))

    client.upload_points(
        TEXT_COLLECTION,
        [models.PointStruct(id=i, vector=text_vecs[i], payload={"source": d.page_content}) for i, d in enumerate(text_docs)],
    )

    client.upload_points(
        IMAGE_COLLECTION,
        [models.PointStruct(id=i, vector=image_vecs[i], payload={"image": Path(d.page_content).name}) for i, d in enumerate(image_docs)],
    )

    return client, embedder


def retrieve_by_text(client: QdrantClient, embedder, query: str, top_k: int = 3):
    q_vec = embedder.get_text_embedding(query)
    res = client.search(collection_name=IMAGE_COLLECTION, query_vector=q_vec, limit=top_k)
    return [point.payload["image"] for point in res]


def retrieve_by_image(client: QdrantClient, embedder, image_path: str, top_k: int = 3):
    q_vec = embedder.get_image_embedding(image_path)
    res = client.search(collection_name=TEXT_COLLECTION, query_vector=q_vec, limit=top_k)
    return [point.payload["source"] for point in res]


### generator.py ###

from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


def init_generator():
    llm = ChatOpenAI(temperature=0.5, model="gpt-3.5-turbo")
    prompt = PromptTemplate(
        input_variables=["query", "context"],
        template="""You are an assistant. Based on the following query and context, provide a relevant and coherent answer.

Query: {query}
Context:
{context}

Answer:"""
    )
    return LLMChain(llm=llm, prompt=prompt)


def generate_response(llm_chain, query: str, retrieved: List[str]) -> str:
    context = "\n".join(retrieved)
    return llm_chain.run({"query": query, "context": context})


### app.py ###

import streamlit as st
from retriever import build_vectorstores, retrieve_by_text, retrieve_by_image
from generator import init_generator, generate_response

st.title("Multimodal Retrieval + Generation Demo")

client, embedder = build_vectorstores()
llm_chain = init_generator()

mode = st.radio("Select mode:", ["Text to Image + Generation", "Image to Text + Generation"])

if mode == "Text to Image + Generation":
    query = st.text_input("Enter your text query:")
    if query:
        images = retrieve_by_text(client, embedder, query)
        st.subheader("Top Image Matches:")
        for img in images:
            st.image(f"data/images/{img}", use_column_width=True)

        response = generate_response(llm_chain, query, images)
        st.subheader("Generated Response:")
        st.write(response)

elif mode == "Image to Text + Generation":
    uploaded_img = st.file_uploader("Upload an image:", type=["png", "jpg", "jpeg"])
    if uploaded_img:
        with open("temp_query.jpg", "wb") as f:
            f.write(uploaded_img.read())
        st.image("temp_query.jpg", use_column_width=True)

        text_chunks = retrieve_by_image(client, embedder, "temp_query.jpg")
        st.subheader("Top Text Matches:")
        for chunk in text_chunks:
            st.write(chunk)

        response = generate_response(llm_chain, "Describe this image", text_chunks)
        st.subheader("Generated Description:")
        st.write(response)


### loaders.py ###

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import os


def load_pdfs_and_texts(folder_path: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = []
    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)
        if fname.lower().endswith(".pdf"):
            for d in splitter.split_documents(PyPDFLoader(fpath).load()):
                docs.append(d)
        elif fname.lower().endswith(".txt"):
            with open(fpath, "r", encoding="utf-8") as f:
                for chunk in splitter.split_text(f.read()):
                    docs.append(Document(page_content=chunk, metadata={"source": fname}))
    return docs


def load_images(folder_path: str):
    return [
        Document(page_content=os.path.join(folder_path, f), metadata={"image": f})
        for f in os.listdir(folder_path)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
    ]


### embedding_utils.py ###

from llama_index.embeddings.huggingface import HuggingFaceEmbedding

_MODEL_ID = "openai/clip-vit-base-patch32"

def get_mm_embedder(device: str = "cpu"):
    return HuggingFaceEmbedding(model_name=_MODEL_ID, device=device, trust_remote_code=True)

#### run_once.py ######

from retriever import build_vectorstores

if __name__ == "__main__":
    build_vectorstores()
    print("✅ Embeddings created and vector stores successfully loaded into Qdrant.")

