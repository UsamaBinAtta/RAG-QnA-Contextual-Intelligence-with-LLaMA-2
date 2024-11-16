from llama_index.core import Settings

def configure_index_settings(llm, embed_model, chunk_size=1024):
    Settings.chunk_size = chunk_size
    Settings.llm = llm
    Settings.embed_model = embed_model