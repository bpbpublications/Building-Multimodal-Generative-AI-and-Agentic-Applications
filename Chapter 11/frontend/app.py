import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from dotenv import load_dotenv
from langchain.schema import HumanMessage

from rag.ollama_llm import get_llm
from rag.graph_workflow import graph
from rag.voice import listen_from_microphone, speak_text

# Load .env file
load_dotenv()

st.set_page_config(page_title="🧠 RAG Chatbot with Voice", layout="wide")
st.title("🧠 Multimodal RAG Chatbot (PDF + Web + Voice)")

# Choose input mode
input_mode = st.radio("Choose input method:", ["Keyboard", "Voice"], horizontal=True)

query = ""

# ✅ Keyboard input
if input_mode == "Keyboard":
    query = st.text_input("Type your question:")

    if query.strip() and st.button("Ask"):
        with st.spinner("Thinking..."):
            state = graph.invoke({
                "question": query,
                "generation": "",
                "web_search": "",
                "max_retries": 2,
                "answers": 0,
                "loop_step": 0,
                "documents": []
            })

            docs = state["documents"]
            source = "web" if state["web_search"] == "Yes" else "pdf"
            context = "\n\n".join(doc.page_content for doc in docs)

            prefix = "According to the web:" if source == "web" else "According to the file provided:"
            prompt = f"""{prefix}

You are a helpful assistant. Use ONLY the information in the CONTEXT below to answer the QUESTION.

CONTEXT:
{context}

QUESTION:
{query}

If the context does not contain the answer, reply exactly with: "I don’t know based on the context."

Answer:"""

            llm = get_llm()
            response = llm.invoke([HumanMessage(content=prompt)])
            final_answer = response.content.strip()

            st.markdown(f"### 🤖 Response:\n{final_answer}")
            st.caption(f"🔎 Source: **{source.upper()}**")
            speak_text(final_answer)

# ✅ Voice input
elif input_mode == "Voice":
    if st.button("🎤 Speak your question"):
        try:
            with st.spinner("Listening..."):
                query = listen_from_microphone()
                st.success(f"🗣️ You said: {query}")

            with st.spinner("Thinking..."):
                state = graph.invoke({
                    "question": query,
                    "generation": "",
                    "web_search": "",
                    "max_retries": 2,
                    "answers": 0,
                    "loop_step": 0,
                    "documents": []
                })

                docs = state["documents"]
                source = "web" if state["web_search"] == "Yes" else "pdf"
                context = "\n\n".join(doc.page_content for doc in docs)

                prefix = "According to the web:" if source == "web" else "According to the file provided:"
                prompt = f"""{prefix}

You are a helpful assistant. Use ONLY the information in the CONTEXT below to answer the QUESTION.

CONTEXT:
{context}

QUESTION:
{query}

If the context does not contain the answer, reply exactly with: "I don’t know based on the context."

Answer:"""

                llm = get_llm()
                response = llm.invoke([HumanMessage(content=prompt)])
                final_answer = response.content.strip()

                st.markdown(f"### 🤖 Response:\n{final_answer}")
                st.caption(f"🔎 Source: **{source.upper()}**")
                speak_text(final_answer)

        except Exception as e:
            st.error(f"❌ Voice error: {e}")
