from langchain_ollama import OllamaLLM

ollama_model = OllamaLLM(model="mistral")

def evaluate_faithfulness_with_ollama(question, context, answer):
    prompt = f"""You are a helpful AI assistant evaluating the FAITHFULNESS of an answer.

Question:
{question}

Retrieved Context:
{context}

Answer:
{answer}

Rate how faithfully the answer is supported by the retrieved context.
Respond ONLY with a number from 1 (hallucinated) to 5 (fully grounded)."""
    response = ollama_model.invoke(prompt).strip()
    try:
        return int(response[0])  # get the first digit as score
    except:
        return 0  # fallback

def evaluate_relevance_with_ollama(question, answer):
    prompt = f"""You are a helpful AI assistant evaluating the RELEVANCE of an answer.

Question:
{question}

Answer:
{answer}

How relevant is this answer to the question? 
Respond ONLY with a number from 1 (not relevant) to 5 (fully relevant)."""
    response = ollama_model.invoke(prompt).strip()
    try:
        return int(response[0])
    except:
        return 0
