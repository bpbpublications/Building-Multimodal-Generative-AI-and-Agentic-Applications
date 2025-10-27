import json
from typing_extensions import TypedDict
from typing import List, Annotated
import operator
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langchain.schema import Document, HumanMessage, SystemMessage

from rag.prompts import rag_prompt
from rag.ollama_llm import get_llm
from rag.embeddings import get_embeddings
from rag.vectorstore import create_vectorstore
from rag.loaders import load_pdfs
from rag.tavily_search import search_tavily
from rag.router import route_question_and_get_source

load_dotenv()

llm = get_llm()
docs = load_pdfs("data/documents")
retriever = create_vectorstore(docs, get_embeddings()).as_retriever(search_kwargs={"k": min(3, len(docs))})


def format_docs(docs): return "\n\n".join(doc.page_content for doc in docs)


class GraphState(TypedDict):
    question: str
    generation: str
    web_search: str
    max_retries: int
    answers: int
    loop_step: Annotated[int, operator.add]
    documents: List[Document]


def retrieve(state):
    return {
        "documents": retriever.invoke(state["question"]),
        "web_search": "No"
    }


def web_search(state):
    docs = search_tavily(state["question"])
    return {
        "documents": docs,
        "web_search": "Yes"
    }


def generate(state):
    context = format_docs(state["documents"])
    question = state["question"]
    web_flag = state.get("web_search", "No")

    # ✅ Summarize raw Tavily results if source is web
    if web_flag == "Yes":
        summary_prompt = f"""Summarize the following search results in 2–3 well-written sentences. 
Avoid listing raw data or metrics. Write it as if you're explaining to a user:

{context}
"""
        summary = llm.invoke([HumanMessage(content=summary_prompt)])
        context = summary.content.strip()  # Replace context with summarized version

    # ✅ Now create the final RAG prompt
    prompt = rag_prompt.format(context=context, question=question)
    result = llm.invoke([HumanMessage(content=prompt)])

    return {
        "generation": result,
        "loop_step": state["loop_step"] + 1
    }



def route_question(state):
    route = route_question_and_get_source(state["question"])
    return "websearch" if route == "web" else "retrieve"


def grade_documents(state):
    # This node is just a passthrough now
    return {
        "documents": state["documents"],
        "web_search": state.get("web_search", "No")
    }


def decide_to_generate(state):
    return "generate" if state["web_search"] == "No" else "websearch"


def grade_generation_v_documents_and_question(state):
    return "useful"


def build_graph():
    workflow = StateGraph(GraphState)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate", generate)
    workflow.add_node("websearch", web_search)
    workflow.add_node("grade_documents", grade_documents)

    workflow.set_conditional_entry_point(route_question, {
        "websearch": "websearch",
        "retrieve": "retrieve"
    })

    workflow.add_edge("retrieve", "grade_documents")

    workflow.add_conditional_edges("grade_documents", decide_to_generate, {
        "generate": "generate",
        "websearch": "websearch"
    })

    workflow.add_conditional_edges("generate", grade_generation_v_documents_and_question, {
        "useful": END,
        "not useful": "websearch",
        "not supported": "generate",
        "max retries": END
    })

    workflow.add_edge("websearch", "generate")

    return workflow.compile()


graph = build_graph()
