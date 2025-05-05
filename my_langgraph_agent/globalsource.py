from .init_env import env
from my_langgraph_agent.vector_store import ChromaStore

from langchain_openai import OpenAIEmbeddings
from transformers import AutoTokenizer # type: ignore
from langchain_openai import ChatOpenAI
from openai import OpenAI

class _Resource:
    MIN_NUM_PREDICT = 4*1024
    def __init__(self):
        extra_body = {
            "chat_template_kwargs": {"enable_thinking": True}
        }
        self.main_model = ChatOpenAI(
            model=env.get('MAIN_MODEL_NAME'),
            base_url=env.get('BASE_URL'),
            api_key=env.get('OPENAI_API_KEY'),
            extra_body=extra_body
        )
        self.openai_model = OpenAI(
            api_key=env.get("OPENAI_API_KEY"), base_url=env.get("BASE_URL")
        )
        self._Chroma = ChromaStore()
        self.embed = OpenAIEmbeddings(
            base_url=env.get("EMBEDER_URL"),
            model=env.get("LOCAL_EMBED_TOKENIZER_PATH"),
            embedding_ctx_length=8100,
            api_key=env.get("EMBEDER_API_KEY"),
            tiktoken_enabled=False,
        )
        
        self.main_model_tokenizer = AutoTokenizer.from_pretrained(
            env.get("LOCAL_TOKENIZER_PATH"), local_files_only=True
        )

    def get_embed(self):
        return self.embed

    def add_documents(self, documents):
        self._Chroma.add_documents(documents)
        self._Chroma.clean_old_docs()

    def search_documents(self, query, k=2):
        self._Chroma.clean_old_docs()
        return self._Chroma.get_documents(query, k)

    def get_main_model_tokenizer(self):
        return self.main_model_tokenizer

    def get_main_model(self):
        return self.main_model

    def get_openai_model(self):
        return self.openai_model


resource = _Resource()
