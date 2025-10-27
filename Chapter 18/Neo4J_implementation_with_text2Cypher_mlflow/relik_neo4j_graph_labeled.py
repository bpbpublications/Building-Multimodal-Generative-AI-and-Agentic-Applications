# build_full_movie_graph.py

# === SUPPRESS DEPRECATION WARNINGS EARLY ===
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import json
import pandas as pd
from relik import Relik
from relik.inference.data.objects import RelikOutput
from neo4j import GraphDatabase

# === CONFIGURATION ===
CSV_PATH       = "data/Updated_Synthetic_Dataset__500_Rows_.csv"
DESC_PATH      = "data/descriptions.json"
NEO4J_URI      = "bolt://localhost:7687"
NEO4J_USER     = "neo4j"
NEO4J_PASSWORD = "Zonunpuia@3104"  # ← your password

# === CLEAR ENTIRE GRAPH ===
def clear_graph(tx):
    tx.run("MATCH (n) DETACH DELETE n")

# === HELPER: Decide labels for ReLiK relations ===
def get_labels(relation: str):
    r = relation.lower()
    if "acted"    in r: return "Actor",    "Movie"
    if "directed" in r: return "Director", "Movie"
    if "genre"    in r: return "Movie",    "Genre"
    if "available"in r: return "Movie",    "Platform"
    if "describ"  in r: return "Chunk",    "Movie"
    if "likes"    in r or "user" in r: return "User",     "Movie"
    return "Entity", "Entity"

# === STEP 1: UPLOAD STRUCTURED METADATA ===
def upload_structured(tx, title, genre, mood, year, actor, director, platform):
    # Movie node + properties
    tx.run(
        """
        MERGE (m:Movie {name:$title})
        SET m.genre = $genre,
            m.mood  = $mood,
            m.release_year = $year
        """,
        title=title, genre=genre, mood=mood, year=int(year),
    )
    # Genre edge
    tx.run(
        """
        MERGE (g:Genre {name:$genre})
        WITH g
        MATCH (m:Movie {name:$title})
        MERGE (m)-[:HAS_GENRE]->(g)
        """,
        title=title, genre=genre,
    )
    # Actor edge
    tx.run(
        """
        MERGE (a:Actor {name:$actor})
        WITH a
        MATCH (m:Movie {name:$title})
        MERGE (a)-[:ACTED_IN]->(m)
        """,
        title=title, actor=actor,
    )
    # Director edge
    tx.run(
        """
        MERGE (d:Director {name:$director})
        WITH d
        MATCH (m:Movie {name:$title})
        MERGE (d)-[:DIRECTED]->(m)
        """,
        title=title, director=director,
    )
    # Platform edge
    tx.run(
        """
        MERGE (p:Platform {name:$platform})
        WITH p
        MATCH (m:Movie {name:$title})
        MERGE (m)-[:AVAILABLE_ON]->(p)
        """,
        title=title, platform=platform,
    )

# === STEP 2: UPLOAD ReLiK TRIPLETS (FILTERED) ===
def upload_triplets(tx, triplets, known_titles):
    for t in triplets:
        head = t.subject.text.strip()
        rel  = t.label
        tail = t.object.text.strip()
        h_label, t_label = get_labels(rel)

        # Skip any Movie node that isn’t in our CSV
        if (h_label=="Movie" and head.lower() not in known_titles) or \
           (t_label=="Movie" and tail.lower() not in known_titles):
            continue

        rel_type = rel.upper().replace(" ", "_").replace("-", "_")
        tx.run(
            f"""
            MERGE (h:{h_label} {{name:$head}})
            MERGE (t:{t_label} {{name:$tail}})
            MERGE (h)-[:{rel_type}]->(t)
            """,
            head=head, tail=tail,
        )

# === MAIN SCRIPT ===
if __name__ == "__main__":
    # 1. Load CSV & descriptions
    df = pd.read_csv(CSV_PATH)
    with open(DESC_PATH) as f:
        descriptions = json.load(f)

    # Prepare a lowercase set of valid movie titles
    known_titles = set(df["title"].astype(str).str.strip().str.lower())

    # 2. Init ReLiK model
    relik = Relik.from_pretrained("relik-ie/relik-relation-extraction-small")

    # 3. Connect to Neo4j
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session() as session:
        # 4. Clear old graph
        session.write_transaction(clear_graph)
        print("🧹 Cleared existing Neo4j graph.")

        # 5. Upload structured metadata
        for row in df.itertuples(index=False):
            session.write_transaction(
                upload_structured,
                row.title, row.genre, row.mood, row.release_year,
                row.actor, row.director, row.platform
            )
        print(f"✅ Uploaded {len(df)} movies with structured metadata.")

        # 6. Upload filtered ReLiK triplets
        for i, desc in enumerate(descriptions):
            out: RelikOutput = relik(desc)
            if out.triplets:
                session.write_transaction(
                    upload_triplets, out.triplets, known_titles
                )
                print(f"[✓] ReLiK chunk {i}: {len(out.triplets)} triplets")

    driver.close()
    print("🎉 Graph build complete. Ready for querying in Neo4j Browser!")
