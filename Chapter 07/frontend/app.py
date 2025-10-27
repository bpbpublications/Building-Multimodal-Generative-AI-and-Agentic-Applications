import streamlit as st
from PIL import Image
import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from rag.index_builder import build_vectorstores, TEXT_COLLECTION, IMAGE_COLLECTION

@st.cache_resource(show_spinner="Loading vector index...")
def init_system():
    return build_vectorstores()

client, mm_embed = init_system()

st.title("🔍 Multimodal Search Demo (Text ↔ Image)")

option = st.radio("Choose your query type:", ["Text → Image", "Image → Text"])

if option == "Text → Image":
    query = st.text_input("Enter a text prompt to retrieve relevant image:")
    if query:
        st.write(f"Searching for image similar to: *{query}*")
        q_vec = mm_embed.get_text_embedding(query)
        res = client.query_points(
            collection_name=IMAGE_COLLECTION,
            query=q_vec,
            using="image",
            with_payload=["image"],
            limit=1,
        )
        if res and res.points:
            image_file = res.points[0].payload["image"]
            st.image(f"data/images/{image_file}", caption="Top Match", use_column_width=True)
        else:
            st.warning("No image match found.")

elif option == "Image → Text":
    uploaded_img = st.file_uploader("Upload an image to find related text", type=["png", "jpg", "jpeg"])
    if uploaded_img:
        with open("temp_input_image.jpg", "wb") as f:
            f.write(uploaded_img.read())

        st.image("temp_input_image.jpg", caption="Uploaded Image", use_column_width=True)

        img_vec = mm_embed.get_image_embedding("temp_input_image.jpg")
        res = client.query_points(
            collection_name=TEXT_COLLECTION,
            query=img_vec,
            using="text",
            with_payload=["source"],
            limit=1,
        )
        if res and res.points:
            source_text = res.points[0].payload["source"]
            st.success("Top matching text:")
            st.write(source_text)
        else:
            st.warning("No relevant text found.")
