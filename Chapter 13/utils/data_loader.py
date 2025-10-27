from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from vectorstore.metadata_schema import add_metadata_to_chunks
from app.config import DATA_FILES
import pandas as pd

def load_and_chunk_all_data():
    all_chunks = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    for path in DATA_FILES:
        df = pd.read_csv(path)
        text = df.astype(str).apply(lambda row: " | ".join(row), axis=1).tolist()
        docs = [Document(page_content=item, metadata={"source": path}) for item in text]
        docs = add_metadata_to_chunks(docs, source_name=path)
        all_chunks.extend(text_splitter.split_documents(docs))

    return all_chunks