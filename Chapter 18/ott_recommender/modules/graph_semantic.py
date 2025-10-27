from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import faiss
import os

def build_semantic_index(desc_path, index_dir):
    documents = SimpleDirectoryReader(input_files=[desc_path]).load_data()
    parser = SentenceSplitter()
    nodes = parser.get_nodes_from_documents(documents)

    embed_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    index = VectorStoreIndex(nodes, embed_model=embed_model)
    index.storage_context.persist(persist_dir=index_dir)

    # Extract the raw FAISS index object and save manually [has to be donme using LLmaIndex]
    if hasattr(index, "index"):
        os.makedirs(index_dir, exist_ok=True)
        faiss.write_index(index.index, os.path.join(index_dir, "index.faiss"))
