def add_metadata_to_chunks(chunks, source_name):
    for chunk in chunks:
        if not chunk.metadata:
            chunk.metadata = {}
        chunk.metadata["source"] = source_name
    return chunks