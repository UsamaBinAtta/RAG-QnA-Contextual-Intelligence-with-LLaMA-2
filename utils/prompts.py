from llama_index.core.prompts.prompts import SimpleInputPrompt

def define_prompts():
    system_prompt = """
    You are a Q&A assistant. Your goal is to answer questions accurately based on the provided context.
    If the question is unrelated, politely indicate that you answer only context-related questions.
    """
    query_wrapper_prompt = SimpleInputPrompt("<|USER|>{query_str}<|ASSISTANT|>")
    return system_prompt, query_wrapper_prompt