from neo4j import GraphDatabase
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

def run_cypher_and_get_results(cypher_query: str, top_k: int = 5):
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    results = []

    with driver.session() as session:
        records = session.run(cypher_query)
        for record in records:
            results.append(record.data())
            if len(results) >= top_k:
                break

    driver.close()
    return results
