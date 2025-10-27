
import requests

def ollama_completion(prompt, model="llama3"):
    response = requests.post("http://localhost:11434/api/generate", json={
        "model": model,
        "prompt": prompt,
        "stream": False
    })
    return response.json()["response"].strip()

