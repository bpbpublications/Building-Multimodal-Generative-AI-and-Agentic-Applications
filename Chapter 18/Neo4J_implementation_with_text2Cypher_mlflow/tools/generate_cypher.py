from config import T2C_MODEL
from langchain_ollama import OllamaLLM  # Updated to latest adapter

def generate_cypher(question: str, schema: str) -> str:
    prompt = f"""Generate a Cypher query for the Question below:
Use the information about the nodes, relationships, and properties from the Schema section below.
Return only the Cypher query. Use double quotes (\"\") for all string literals, including list values.
####Schema:
{schema}
####Question:
{question}
"""
    model = OllamaLLM(model=T2C_MODEL)
    response = model.invoke(prompt)
    cypher = response.strip()

    # Remove surrounding quotes if present
    if cypher.startswith('"') and cypher.endswith('"'):
        cypher = cypher[1:-1]

    # Convert literal \n to actual newlines
    cypher = cypher.replace("\\n", "\n")

    # Convert single quotes to double quotes
    cypher = cypher.replace("'", '"')

    # Optional debug print
    print(f"\n🔎 Cleaned Cypher Query:\n{cypher}")

    return cypher
