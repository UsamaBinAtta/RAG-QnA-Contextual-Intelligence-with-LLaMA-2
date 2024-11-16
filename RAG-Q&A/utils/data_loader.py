from llama_index.core import SimpleDirectoryReader

def load_documents(data_path):
    reader = SimpleDirectoryReader(data_path)
    documents = reader.load_data()
    print("Documents Loaded: ", documents)
    return documents