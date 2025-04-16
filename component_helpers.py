def strip_think(message: str) -> str:
    if "</think>" in message:
        message = message.split("</think>")[1].lstrip('\n')
    return message

from transformers import AutoTokenizer
import os
tokenizer_path = os.environ.get("LOCAL_TOKENIZER_PATH", "deepseek-ai/DeepSeek-R1")
print(f"Initializing tokenizer from: {tokenizer_path}")
TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_path)

# return ids of embeded documents
# you'd better save this list to remove them later
def embed_message(documents, vector_store, spliter) -> list[str]:
    all_splits = spliter.split_documents(documents)
    return vector_store.add_documents(documents=all_splits)

def init_embed_vector_spliter(embeder_url="http://127.0.0.1:11434", embed_model="bge-m3", persist_directory = None, seperated=True, separators='\n', chunk_size=2000, chunk_overlap=200, add_start_index=True):
    from langchain_ollama import OllamaEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_chroma import Chroma
    import uuid
    embed = OllamaEmbeddings(
        base_url=embeder_url,
        model=embed_model
    )
    collection_id = uuid.uuid4().hex if seperated else "global"
    vector_store = Chroma(
        collection_name=collection_id,
        embedding_function=embed,
        persist_directory=persist_directory
    )
    text_splitter = RecursiveCharacterTextSplitter(
        separators=separators,
        chunk_size=chunk_size,  # chunk size (characters)
        chunk_overlap=chunk_overlap,  # chunk overlap (characters)
        add_start_index=add_start_index,  # track index in original document
    )
    return embed, vector_store, text_splitter, collection_id

from langchain_core.messages import BaseMessage
from typing import List
from langchain_core.messages import convert_to_openai_messages
def tiktoken_counter(messages: List[BaseMessage]) -> int:
    """
    用 AutoTokenizer 准确计算 vllm 认为的 token 数量。For simplicity only supports str Message.contents.
    """
    global TOKENIZER
    openai_messages = convert_to_openai_messages(messages)
    # 根据 vllm 的 /tokenize 接口测试结果，vllm是会 add_generation_prompt 的
    return len(TOKENIZER.apply_chat_template(convert_to_openai_messages(openai_messages), add_generation_prompt=True, tokenize=True))

def raw_tiktoken_counter(messages: list[dict[str: str]]) -> int:
    """
    用 AutoTokenizer 准确计算 vllm 认为的 token 数量。For simplicity only supports str Message.contents.
    """
    global TOKENIZER
    return len(TOKENIZER.apply_chat_template(convert_to_openai_messages(messages), add_generation_prompt=True, tokenize=True))

# You can use this main to test if the result is the same as vllm's /tokenize result
if __name__ == "__main__":
    from langchain_core.messages import convert_to_openai_messages
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="Hello, world!"),
        AIMessage(content="Hello, how can I help you today?")
    ]
    print(len(TOKENIZER.apply_chat_template(convert_to_openai_messages(messages), add_generation_prompt=True, tokenize=True)))