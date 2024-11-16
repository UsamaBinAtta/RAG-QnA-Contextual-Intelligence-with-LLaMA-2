import os

# Directory paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# HuggingFace token for authentication
HUGGINGFACE_TOKEN = "your_huggingface_token_here"

# Model configurations
MODEL_NAME = "usamaatta/Llama-2-7b-chat-CNN-chatbot"
TOKENIZER_NAME = "usamaatta/Llama-2-7b-chat-CNN-chatbot"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

# Other configuration values
CHUNK_SIZE = 1024
MAX_TOKENS = 256
TEMPERATURE = 0.7
DO_SAMPLE = False