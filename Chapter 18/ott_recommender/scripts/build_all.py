from app.config import DATA_FILES
from modules.nlu_converter import generate_nl_descriptions
from modules.graph_structured_local import build_structured_graph
from modules.graph_semantic import build_semantic_index
from modules.vector_store import build_user_profile_index

def run_build_pipeline():
    generate_nl_descriptions(DATA_FILES['catalog'], 'data/descriptions.json')
    build_structured_graph(DATA_FILES['catalog'], 'index/structured_graph.gpickle')
    build_semantic_index('data/descriptions.json', 'index/catalog_faiss')
    build_user_profile_index(DATA_FILES['preferences'], 'index/preferences_faiss')

if __name__ == "__main__":
    run_build_pipeline()