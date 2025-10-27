import os
import requests
from langchain.schema import Document
from dotenv import load_dotenv

# Load API key from .env file (called once at import time)
load_dotenv()

def search_tavily(query: str, max_results: int = 3) -> list[Document]:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY not found in environment variables or .env")

    url = "https://api.tavily.com/search"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {
        "query": query,
        "num_results": max_results
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"Tavily search API request failed: {e}")

    try:
        data = response.json()
        results = data.get("results", [])
    except Exception as e:
        raise ValueError(f"Failed to parse Tavily response: {e}")

    if not results:
        return [Document(page_content="Tavily returned no results.")]

    return [Document(page_content=entry["content"]) for entry in results if "content" in entry]
