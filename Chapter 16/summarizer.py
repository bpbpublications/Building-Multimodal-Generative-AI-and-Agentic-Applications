import requests
from bs4 import BeautifulSoup
import ollama

def fetch_page_snippet(url, max_items=5):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0"
        }
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Try to extract product titles and prices from common patterns
        snippets = []

        for tag in soup.find_all(['a', 'div', 'span'], limit=100):
            text = tag.get_text(strip=True)
            if text and len(text) > 20:
                snippets.append(text)

        return snippets[:max_items]

    except Exception as e:
        return [f"Error fetching {url}: {str(e)}"]

def summarize_snippets(snippets, site_name):
    joined_text = "\\n".join(snippets)
    prompt = (
        f"The following are some product listings from {site_name}:\n\n"
        f"{joined_text}\n\n"
        f"Summarize the main themes or trends in the product listings. "
        f"Include keywords, brands, price trends, and any notable offers."
    )

    response = ollama.chat(
        model="llama3.2:3b-instruct-fp16",
        messages=[{"role": "user", "content": prompt}]
    )
    return response['message']['content'].strip()

def summarize_product_pages(product_listings):
    all_summaries = []
    for listing in product_listings:
        print(f"Fetching from {listing['merchant']}...")
        snippets = fetch_page_snippet(listing['link'])
        summary = summarize_snippets(snippets, listing['merchant'])
        all_summaries.append({
            "store": listing['merchant'],
            "summary": summary
        })
    return all_summaries
