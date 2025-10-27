# core/chroma_index.py

import chromadb

client = chromadb.PersistentClient(path="./global_index_db")
global_index = client.get_or_create_collection("global_index")
row_data_collection = client.get_or_create_collection("row_data")
print("row_data_collection initialized:", row_data_collection)
