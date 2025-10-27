import pandas as pd
import networkx as nx
import pickle
import json

def build_structured_graph(csv_path, output_path=None, description_path='data/descriptions.json'):
    df = pd.read_csv(csv_path)
    with open(description_path) as f:
        descriptions = json.load(f)

    G = nx.MultiDiGraph()

    for i, row in df.iterrows():
        title = row['title']

        # Add Title node with attributes
        G.add_node(title, label="Title",
                   title=title,
                   genre=row.get('genre', 'NA'),
                   mood=row.get('mood', 'NA'),
                   year=row.get('release_year', 'NA'))

        # Add Genre nodes and edges
        for genre in str(row['genre']).split(','):
            genre = genre.strip()
            if genre:
                G.add_node(genre, label="Genre")
                G.add_edge(title, genre, type="HAS_GENRE")

        # Add Actor nodes and edges
        for actor in str(row['actors']).split(','):
            actor = actor.strip()
            if actor:
                G.add_node(actor, label="Actor")
                G.add_edge(actor, title, type="ACTED_IN")

        # Add Platform node and edge
        platform = row['content_type']
        G.add_node(platform, label="Platform")
        G.add_edge(title, platform, type="AVAILABLE_ON")

        # Add Director nodes and edges
        for director in str(row['directors']).split(','):
            director = director.strip()
            if director:
                G.add_node(director, label="Director")
                G.add_edge(director, title, type="DIRECTED")

        # Add Chunk node and edge
        if i < len(descriptions):
            chunk_id = f"chunk_{i}"
            G.add_node(chunk_id, label="Chunk", text=descriptions[i])
            G.add_edge(chunk_id, title, type="DESCRIBES")

    if output_path:
        with open(output_path, 'wb') as f:
            pickle.dump(G, f)
        nx.write_graphml(G, output_path.replace('.gpickle', '.graphml'))

    return G
