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
