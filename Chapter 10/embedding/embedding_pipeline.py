# embedding_pipeline.py

from qdrant_client import QdrantClient, models
from fastembed import LateInteractionTextEmbedding
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os
import torch
import uuid  # ✅ NEW

# ==== CONFIGURATION ====
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "multimodal_multivector"
DENSE_MODEL_NAME = "BAAI/bge-small-en"
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
COLBERT_MODEL_NAME = "colbert-ir/colbertv2.0"

# ==== INITIALIZATION ====
client = QdrantClient(QDRANT_URL)
dense_model = SentenceTransformer(DENSE_MODEL_NAME)
clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
colbert_embedder = LateInteractionTextEmbedding(model_name=COLBERT_MODEL_NAME)

# ==== EMBEDDING FUNCTIONS ====
def generate_text_embedding(text):
    return dense_model.encode(text).tolist()

def generate_colbert_embedding(text):
    return list(colbert_embedder.embed([text]))[0]  # convert generator to list

def generate_image_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        image_features = clip_model.get_image_features(**inputs)
    return image_features[0].cpu().tolist()

# ==== ADAPTIVE REFRESH PIPELINE ====
def refresh_embeddings(image_folder, text_folder):
    points = []
    for fname in os.listdir(text_folder):
        if not fname.endswith(".txt"):
            continue

        file_base = os.path.splitext(fname)[0]
        text_path = os.path.join(text_folder, fname)
        image_path = os.path.join(image_folder, file_base + ".jpg")

        if not os.path.exists(image_path):
            continue  # Skip if corresponding image is missing

        with open(text_path, 'r') as f:
            text = f.read()

        dense_vec = generate_text_embedding(text)
        colbert_vecs = generate_colbert_embedding(text)
        image_vec = generate_image_embedding(image_path)

        points.append(models.PointStruct(
            id=str(uuid.uuid4()),  # ✅ VALID UUID
            vector={
                "dense_text": dense_vec,
                "colbert_text": colbert_vecs,
                "image": image_vec
            },
            payload={
                "filename": fname,
                "text": text
            }
        ))

    client.upsert(collection_name=COLLECTION_NAME, points=points)

# ==== COLLECTION SETUP ====
def create_collection():
    try:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                "dense_text": models.VectorParams(
                    size=384,
                    distance=models.Distance.COSINE
                ),
                "colbert_text": models.VectorParams(
                    size=128,
                    distance=models.Distance.COSINE,
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM
                    ),
                    hnsw_config=models.HnswConfigDiff(m=0)
                ),
                "image": models.VectorParams(
                    size=512,
                    distance=models.Distance.COSINE
                )
            }
        )
    except Exception as e:
        if "already exists" in str(e):
            print(f"[i] Collection '{COLLECTION_NAME}' already exists. Skipping creation.")
        else:
            raise

if __name__ == "__main__":
    create_collection()
    refresh_embeddings(image_folder="data/images", text_folder="data/documents")
