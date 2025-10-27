from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import SOURCE_DOC
from vectorstore.metadata_schema import add_metadata_to_chunks
import os

def load_and_chunk_pdf():
    loader = PyPDFLoader(SOURCE_DOC)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = text_splitter.split_documents(documents)

    # Add metadata with document name
    source_name = os.path.basename(SOURCE_DOC)
    chunks = add_metadata_to_chunks(chunks, source_name)

    return chunks
