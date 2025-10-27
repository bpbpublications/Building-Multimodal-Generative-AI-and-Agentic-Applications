# ott_recommender/app/query_agent.py

from langchain.agents import initialize_agent, Tool
from langchain_community.llms import Ollama
from app.prompts import SYSTEM_PROMPT

import pickle
import numpy as np
import json
import faiss
import networkx as nx
from sentence_transformers import SentenceTransformer
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from langchain_community.embeddings import HuggingFaceEmbeddings
from difflib import SequenceMatcher

# Load semantic content
with open("data/descriptions.json") as f:
    DESCRIPTIONS = json.load(f)

# Load user profiles and FAISS index
with open("data/User_Preference_Profiles.csv") as f:
    lines = f.readlines()[1:]
    PROFILES = [line.strip().split(",", 1)[1] for line in lines]
PROFILE_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
PROFILE_INDEX = faiss.read_index("index/preferences_faiss/preferences.index")

# Load structured graph
GRAPH = pickle.load(open("index/structured_graph.gpickle", "rb"))

# Load vector index with higher top_k
storage_context = StorageContext.from_defaults(persist_dir="index/catalog_faiss")
embed_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
VECTOR_INDEX = load_index_from_storage(storage_context, embed_model=embed_model)
RETRIEVER = VECTOR_INDEX.as_retriever(similarity_top_k=10)


def enrich_query_with_profile(query):
    query_vec = PROFILE_MODEL.encode([query])
    _, I = PROFILE_INDEX.search(np.array(query_vec), k=1)
    matched_profile = PROFILES[I[0][0]]
    return f"{query}. My preferences are: {matched_profile}"


def vector_search_tool(input):
    enriched_query = enrich_query_with_profile(input)

    # Set similarity_top_k high enough to get more results
    results = RETRIEVER.retrieve(enriched_query)
    if not results:
        return "No semantic matches found."

    top_results = []
    for i, r in enumerate(results[:5], 1):
        meta = r.metadata
        title = meta.get("Title", "Unknown")
        year = meta.get("Release Year", "NA")
        genre = meta.get("Genre", "NA")
        mood = meta.get("Mood", "NA")
        top_results.append(f"#{i}. {title} ({year}) - Genre: {genre}, Mood: {mood}")

    return "\n".join(top_results)



from difflib import SequenceMatcher

def graph_search_tool(input):
    results = []
    input_lower = input.lower()
    for node in GRAPH.nodes(data=True):
        label = node[1].get("label", "")
        title = node[0].lower()
        if label.lower() == "title":
            score = SequenceMatcher(None, input_lower, title).ratio()
            if score > 0.3:  # tune this threshold
                genre = node[1].get("genre", "NA")
                year = node[1].get("year", "NA")
                mood = node[1].get("mood", "NA")
                results.append((score, f"{node[0]} ({year}) - Genre: {genre}, Mood: {mood} [Source: Graph]"))

    if not results:
        return "No structured titles matched your input."

    # Sort by relevance
    top_results = sorted(results, reverse=True)[:5]
    return "\n".join([r[1] for r in top_results])


def hybrid_retriever(query, top_k=5):
    enriched_query = enrich_query_with_profile(query)

    # Vector search
    vector_results = RETRIEVER.retrieve(enriched_query)
    vector_titles = [
        {
            "title": r.metadata.get("Title", "Unknown"),
            "year": r.metadata.get("Release Year", "NA"),
            "genre": r.metadata.get("Genre", "NA"),
            "mood": r.metadata.get("Mood", "NA"),
            "source": "Vector"
        }
        for r in vector_results
    ]

    # Graph search (fuzzy fallback)
    graph_titles = []
    input_lower = query.lower()
    from difflib import SequenceMatcher

    for node in GRAPH.nodes(data=True):
        label = node[1].get("label", "")
        title = node[0]
        if label.lower() == "title":
            score = SequenceMatcher(None, input_lower, title.lower()).ratio()
            if score > 0.3:
                graph_titles.append({
                    "title": title,
                    "year": node[1].get("year", "NA"),
                    "genre": node[1].get("genre", "NA"),
                    "mood": node[1].get("mood", "NA"),
                    "source": "Graph"
                })

    # Merge & deduplicate by title
    seen_titles = set()
    combined_results = []

    for item in vector_titles + graph_titles:
        if item["title"] not in seen_titles:
            seen_titles.add(item["title"])
            combined_results.append(item)

    if not combined_results:
        return "No relevant results found in hybrid retrieval."

    top_results = combined_results[:top_k]
    formatted = [
        f"#{i+1}. {r['title']} ({r['year']}) - Genre: {r['genre']}, Mood: {r['mood']} [Source: {r['source']}]"
        for i, r in enumerate(top_results)
    ]
    return "\n".join(formatted)



def hybrid_tool(input):
    return hybrid_retriever(input, top_k=5)


def build_agent():
    tools = [
        Tool(
            name="HybridRetriever",
            func=hybrid_tool,
            description="Combined search over structured graph and semantic vector data"
        ),
        Tool(
            name="GraphSearch",
            func=graph_search_tool,
            description="Structured catalog search"
        ),
        Tool(
            name="VectorSearch",
            func=vector_search_tool,
            description="Semantic search over content descriptions"
        )
    ]

    llm = Ollama(model="mistral", system=SYSTEM_PROMPT)
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent="zero-shot-react-description",
        verbose=True,
        handle_parsing_errors=True
    )
    return agent
