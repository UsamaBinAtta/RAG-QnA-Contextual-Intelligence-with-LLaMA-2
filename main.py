from utils.data_loader import load_documents
from utils.llm_initializer import initialize_llm, initialize_embeddings
from utils.prompts import define_prompts
from utils.settings import configure_index_settings
from utils.index_manager import create_index, query_index

data_path = "/content/drive/MyDrive/data"

def initialize_rag_pipeline(data_path):
    documents = load_documents(data_path)
    system_prompt, query_wrapper_prompt = define_prompts()
    llm = initialize_llm(system_prompt, query_wrapper_prompt)
    embed_model = initialize_embeddings()
    configure_index_settings(llm, embed_model)
    index = create_index(documents)
    return index

if __name__ == "__main__":
    index = initialize_rag_pipeline(data_path)
    query = "What role does the pooling layer play in CNNs?"
    response = query_index(index, query)
    print(response)