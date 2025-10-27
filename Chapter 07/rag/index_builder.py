from pathlib import Path
from typing import List
from rag.embedding_utils import get_mm_embedder
from qdrant_client import QdrantClient, models
from rag.loaders import load_pdfs_and_texts, load_images
from langchain.schema import Document
from numpy.linalg import norm

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
