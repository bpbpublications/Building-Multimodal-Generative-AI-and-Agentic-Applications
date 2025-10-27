def add_metadata_to_chunks(chunks, source_name):
    """
    Adds document name as metadata to each chunk.

    Args:
        chunks (List[Document]): List of LangChain Document objects.
        source_name (str): Name of the source document (e.g., PDF file name).

    Returns:
        List[Document]: Updated chunks with source metadata.
    """
    for chunk in chunks:
        if not chunk.metadata:
            chunk.metadata = {}
        chunk.metadata["source"] = source_name
    return chunks
