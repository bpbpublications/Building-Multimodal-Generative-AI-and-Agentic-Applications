### retriever.py or index_builder.py ###

... [REMAINS UNCHANGED] ...

### generator.py ###

... [REMAINS UNCHANGED] ...

### grader.py ###

from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


def init_grader():
    llm = ChatOpenAI(temperature=0.3, model="gpt-3.5-turbo")
    prompt = PromptTemplate(
        input_variables=["query", "context", "response"],
        template="""Evaluate the quality of the following generated response.

Query: {query}
Context: {context}
Response: {response}

Give a score from 1 to 5 and explain why.

Score and Justification:"""
    )
    return LLMChain(llm=llm, prompt=prompt)


def grade_response(grader_chain, query: str, context: str, response: str):
    return grader_chain.run({
        "query": query,
        "context": context,
        "response": response
    })


### app.py ###

import streamlit as st
from retriever import build_vectorstores, retrieve_by_text, retrieve_by_image
from generator import init_generator, generate_response
from grader import init_grader, grade_response

st.title("Multimodal Retrieval + Generation + Grading Demo")

client, embedder = build_vectorstores()
llm_chain = init_generator()
grader_chain = init_grader()

mode = st.radio("Select mode:", ["Text to Image + Generation", "Image to Text + Generation"])

if mode == "Text to Image + Generation":
    query = st.text_input("Enter your text query:")
    if query:
        images = retrieve_by_text(client, embedder, query)
        st.subheader("Top Image Matches:")
        for img in images:
            st.image(f"data/images/{img}", use_column_width=True)

        response = generate_response(llm_chain, query, images)
        st.subheader("Generated Response:")
        st.write(response)

        grade = grade_response(grader_chain, query, "\n".join(images), response)
        st.subheader("Grading Evaluation:")
        st.write(grade)

elif mode == "Image to Text + Generation":
    uploaded_img = st.file_uploader("Upload an image:", type=["png", "jpg", "jpeg"])
    if uploaded_img:
        with open("temp_query.jpg", "wb") as f:
            f.write(uploaded_img.read())
        st.image("temp_query.jpg", use_column_width=True)

        text_chunks = retrieve_by_image(client, embedder, "temp_query.jpg")
        st.subheader("Top Text Matches:")
        for chunk in text_chunks:
            st.write(chunk)

        response = generate_response(llm_chain, "Describe this image", text_chunks)
        st.subheader("Generated Description:")
        st.write(response)

        grade = grade_response(grader_chain, "Describe this image", "\n".join(text_chunks), response)
        st.subheader("Grading Evaluation:")
        st.write(grade)


### loaders.py ###

... [REMAINS UNCHANGED] ...

### embedding_utils.py ###

... [REMAINS UNCHANGED] ...

#### run_once.py ######

... [REMAINS UNCHANGED] ...
