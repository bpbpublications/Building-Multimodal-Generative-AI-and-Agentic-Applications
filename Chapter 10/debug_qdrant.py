from qdrant_client import QdrantClient

client = QdrantClient("http://localhost:6333")
COLLECTION_NAME = "multimodal_multivector"

# Fetch one record
result, _ = client.scroll(
    collection_name=COLLECTION_NAME,
    limit=1,
    with_vectors=True
)

point = result[0]

print("\n✅ Fields present in the vector store:")
for key in point.vector.keys():
    print(f" - {key}")

# Check if ColBERT multivector is valid
colbert_vecs = point.vector.get("colbert_text")
if isinstance(colbert_vecs, list) and len(colbert_vecs) > 1:
    print(f"\n✅ colbert_text is multivector with {len(colbert_vecs)} token-level vectors.")
else:
    print(f"\n❌ colbert_text is NOT a multivector or is empty.")

# Check image vector
image_vec = point.vector.get("image")
if isinstance(image_vec, list) and len(image_vec) == 512:
    print("✅ image embedding exists and has 512 dimensions.")
else:
    print("❌ image embedding missing or incorrect dimensions.")
