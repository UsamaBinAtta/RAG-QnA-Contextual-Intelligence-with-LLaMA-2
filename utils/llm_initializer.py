from llama_index.llms.huggingface import HuggingFaceLLM
from langchain.embeddings import HuggingFaceEmbeddings
from llama_index.embeddings.langchain import LangchainEmbedding
import torch

def initialize_llm(system_prompt, query_wrapper_prompt):
    llm = HuggingFaceLLM(
        context_window=4096,
        max_new_tokens=256,
        generate_kwargs={"temperature": 0.7, "do_sample": False},
        system_prompt=system_prompt,
        query_wrapper_prompt=query_wrapper_prompt,
        tokenizer_name="usamaatta/Llama-2-7b-chat-CNN-chatbot",
        model_name="usamaatta/Llama-2-7b-chat-CNN-chatbot",
        device_map="auto",
        model_kwargs={
            "torch_dtype": torch.float16,
            "load_in_8bit": True
        }
    )
    return llm

def initialize_embeddings():
    embed_model = LangchainEmbedding(
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    )
    return embed_model