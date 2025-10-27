import os
import chromadb
from chromadb.utils import embedding_functions
from .embedding_utils import embed_text_ollama, embed_image_ollama
from .config import *
from .loaders import load_text_documents, load_image_paths

def build_index():
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

    # Text Collection
    if CHROMA_TEXT_COLLECTION in [c.name for c in client.list_collections()]:
        client.delete_collection(name=CHROMA_TEXT_COLLECTION)
    text_collection = client.create_collection(name=CHROMA_TEXT_COLLECTION)

    texts = load_text_documents(TEXT_FOLDER)
    for idx, (fname, content) in enumerate(texts.items()):
        emb = embed_text_ollama(content)
        text_collection.add(documents=[content], embeddings=[emb], ids=[str(idx)], metadatas=[{"file": fname}])

    # Image Collection
    if CHROMA_IMAGE_COLLECTION in [c.name for c in client.list_collections()]:
        client.delete_collection(name=CHROMA_IMAGE_COLLECTION)
    print("📦 Creating image collection...")
    image_collection = client.create_collection(name=CHROMA_IMAGE_COLLECTION)
    print("✅ Collections:", [c.name for c in client.list_collections()])

    images = load_image_paths(IMAGE_FOLDER)
    print(f"📸 Found {len(images)} images:", images)
    for idx, path in enumerate(images):
        print(f"🔄 Embedding image: {path}")
        emb = embed_image_ollama(path)
        image_collection.add(documents=[""], embeddings=[emb], ids=[str(idx)], metadatas=[{"file": os.path.basename(path)}])