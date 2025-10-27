
import ollama

def generate_search_query(ocr_text):
    prompt = (
        f"The following product text was extracted from an image:\n\n"
        f"{ocr_text}\n\n"
        f"Generate a realistic Flipkart search query a user would type to find alternatives from known brands "
        f"in the same price range. Just return the query string — no formatting, no explanation, no punctuation.\n\n"
        f"Example: wired headphones with mic under 800"
    )

    response = ollama.chat(
        model="llama3.2:3b-instruct-fp16",
        messages=[{"role": "user", "content": prompt}]
    )
    return response['message']['content'].strip()