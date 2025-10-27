# run_once.py

from rag.loaders import load_pdfs
from rag.embeddings import get_embeddings
from rag.vectorstore import create_vectorstore

def main():
    print("📚 Loading documents from data/documents...")
    docs = load_pdfs("data/documents")
    print(f"✅ Loaded {len(docs)} documents.")

    print("🧠 Creating vectorstore...")
    vectorstore = create_vectorstore(docs, get_embeddings())

    print(f"✅ Embedded {len(docs)} documents into vectorstore.")

if __name__ == "__main__":
    main()
