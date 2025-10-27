import os

# Neo4j Config (for Neo4j Desktop)
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")  # No hardcoded default for safety

# LLM Model Configs (Ollama)
T2C_MODEL = os.getenv("T2C_MODEL", "ed-neo4j/t2c-gemma3-4b-it-q8_0-35k")
SUMMARIZER_MODEL = os.getenv("SUMMARIZER_MODEL", "mistral")

# Optional: prompt for password if not provided
if NEO4J_PASSWORD is None:
    import getpass
    NEO4J_PASSWORD = getpass.getpass("Enter Neo4j password: ")