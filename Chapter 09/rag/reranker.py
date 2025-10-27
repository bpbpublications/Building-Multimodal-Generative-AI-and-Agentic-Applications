from sentence_transformers import CrossEncoder

cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def rerank(query, metadatas):
    pairs = [(query, doc.get("file", "")) for doc in metadatas]
    scores = cross_encoder.predict(pairs)
    ranked = sorted(zip(metadatas, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked]