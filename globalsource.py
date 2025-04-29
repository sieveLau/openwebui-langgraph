from time import sleep
from init_env import env

from langchain_openai import OpenAIEmbeddings
from transformers import AutoTokenizer
from langchain_text_splitters import TokenTextSplitter
from langchain.chat_models import init_chat_model
from openai import OpenAI
import redis
from langchain_redis import RedisConfig, RedisVectorStore
from datetime import datetime


class _Resource:
    GLOBAL_VOLATILE_INDEX_NAME = "volatile_storage"
    MINIMUM_SCHEMA = [
        {"name": "source", "type": "text"},
        {"name": "title", "type": "text"},
    ]

    def __init__(self):
        self.NUM_PREDICT = 6 * 1024
        self.main_model = init_chat_model(
            model=env.get("MAIN_MODEL_NAME"),
            model_provider="openai",
            base_url=env.get("BASE_URL"),
            max_tokens=self.NUM_PREDICT,
        )
        self.openai_model = OpenAI(
            api_key=env.get("OPENAI_API_KEY"), base_url=env.get("BASE_URL")
        )

        self.embed = OpenAIEmbeddings(
            base_url=env.get("EMBEDER_URL"),
            model=env.get("LOCAL_EMBED_TOKENIZER_PATH"),
            embedding_ctx_length=8100,
            api_key=env.get("EMBEDER_API_KEY"),
            tiktoken_enabled=False,
        )

        self.text_splitter = TokenTextSplitter.from_huggingface_tokenizer(
            chunk_size=1000,  # chunk size (characters)
            chunk_overlap=0,  # chunk overlap (characters)
            tokenizer=AutoTokenizer.from_pretrained(
                env.get("LOCAL_TOKENIZER_PATH"), local_files_only=True
            ),
            add_start_index=True,  # track index in original document
        )

        self.main_model_tokenizer = AutoTokenizer.from_pretrained(
            env.get("LOCAL_TOKENIZER_PATH"), local_files_only=True
        )

        self._redis_client = redis.from_url(env.get("REDIS_URL"))
        # on this machine, redis will be up after a few seconds
        while not self._redis_client.ping():
            sleep(5)

        config = RedisConfig(
            index_name=self.GLOBAL_VOLATILE_INDEX_NAME,
            redis_client=self._redis_client,
            metadata_schema=[
                {"name": "title", "type": "text"},
                {"name": "source", "type": "text"},
                {"name": "add_date", "type": "text"},
            ],
        )

        self.vector_store = RedisVectorStore(
            self.embed, config=config, ttl=7 * 24 * 3600
        )

    def get_embed(self):
        return self.embed

    def get_text_splitter(self):
        return self.text_splitter

    def get_vector_store(self):
        return self.vector_store

    def add_documents(self, documents):
        now = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S UTC%z")
        for doc in documents:
            if doc.metadata is not None:
                doc.metadata["add_date"] = now
            else:
                doc.metadata = {
                    "title": "unknown",
                    "source": "unknown",
                    "add_date": now,
                }
        all_splits = self.text_splitter.split_documents(documents)

        ids = self.vector_store.add_documents(all_splits)
        return ids

    def search_documents(self, query, k=2):
        docs, scores = zip(*self.vector_store.similarity_search_with_score(query, k))
        for doc, score in zip(docs, scores):
            doc.metadata["score"] = score
        return docs

    def get_retriver(self, k: int):
        return self.vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": k}
        )

    def get_main_model_tokenizer(self):
        return self.main_model_tokenizer

    def get_main_model(self):
        return self.main_model

    def get_openai_model(self):
        return self.openai_model


resource = _Resource()
