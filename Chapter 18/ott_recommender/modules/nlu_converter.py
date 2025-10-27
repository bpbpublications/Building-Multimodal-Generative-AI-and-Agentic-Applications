# ott_recommender/modules/nlu_converter.py

import pandas as pd
import json

def convert_row_to_nl(row):
    return (
        f"Title: {row['title']}, Genre: {row['genre']}, "
        f"Actors: {row['actors']}, Mood: {row['mood']}, "
        f"Director: {row['directors']}, Platform: {row['content_type']}, "
        f"Release Year: {row['release_year']}"
    )

def generate_nl_descriptions(csv_path, output_path):
    df = pd.read_csv(csv_path)
    descriptions = [convert_row_to_nl(row) for _, row in df.iterrows()]
    with open(output_path, 'w') as f:
        json.dump(descriptions, f, indent=2)
