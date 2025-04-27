from init_env import env

import os
from langchain_openai import OpenAIEmbeddings
from transformers import AutoTokenizer
from langchain_text_splitters import CharacterTextSplitter
from langchain.chat_models import init_chat_model
from openai import OpenAI

class _Resource:
    def __init__(self):
        self.NUM_PREDICT = 6*1024
        self.main_model = init_chat_model(model="deepseek-r1", model_provider="openai", base_url=env.get('BASE_URL'), max_tokens=self.NUM_PREDICT)
        self.openai_model = OpenAI(api_key=env.get('OPENAI_API_KEY'),base_url = env.get('BASE_URL'))

        self.embed = OpenAIEmbeddings(
            base_url=env.get('EMBEDER_URL'),
            model=env.get('LOCAL_EMBED_TOKENIZER_PATH'),
            embedding_ctx_length=8100,
            api_key=env.get('EMBEDER_API_KEY'),
            tiktoken_enabled=False
        )

        self.text_splitter = CharacterTextSplitter.from_huggingface_tokenizer(
            chunk_size=1000,  # chunk size (characters)
            chunk_overlap=0,  # chunk overlap (characters)
            tokenizer=AutoTokenizer.from_pretrained(env.get("LOCAL_TOKENIZER_PATH"),local_files_only=True),
            add_start_index=True  # track index in original document
        )

        self.main_model_tokenizer = AutoTokenizer.from_pretrained(env.get("LOCAL_TOKENIZER_PATH"), local_files_only=True)

    def get_embed(self):
        return self.embed
    def get_text_splitter(self):
        return self.text_splitter
    def get_main_model_tokenizer(self):
        return self.main_model_tokenizer
    def get_main_model(self):
        return self.main_model
    def get_openai_model(self):
        return self.openai_model

resource = _Resource()