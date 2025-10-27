
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def build_user_profile_index(csv_path, output_dir):
    df = pd.read_csv(csv_path)
    print("Available columns:", df.columns.tolist())
    # Auto-detect the first text-like column for embedding
    text_column = None
    for col in df.columns:
        if df[col].dtype == object:
            text_column = col
            break
    if text_column is None:
        raise ValueError("No valid text column found for embedding in user profile CSV.")

    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(df[text_column].tolist())
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    faiss.write_index(index, f"{output_dir}/preferences.index")