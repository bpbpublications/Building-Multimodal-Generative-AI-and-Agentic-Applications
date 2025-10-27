from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import SOURCE_DOCS
from vectorstore.metadata_schema import add_metadata_to_chunks
import os

def load_and_chunk_pdfs():
    all_chunks = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )

    for path in SOURCE_DOCS:
        loader = PyPDFLoader(path)
        documents = loader.load()
        chunks = splitter.split_documents(documents)
        source_name = os.path.basename(path)
        enriched_chunks = add_metadata_to_chunks(chunks, source_name)
        all_chunks.extend(enriched_chunks)

    return all_chunks
