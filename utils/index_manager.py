from llama_index.core import VectorStoreIndex

def create_index(documents):
    index = VectorStoreIndex.from_documents(
        documents,
        llm=Settings.llm,
        embed_model=Settings.embed_model,
        chunk_size=Settings.chunk_size
    )
    return index

def query_index(index, query):
    if index is None:
        raise ValueError("Index not initialized. Please load documents and create the index first.")
    query_engine = index.as_query_engine()
    response = query_engine.query(query)
    return response