import chromadb
from my_langgraph_agent import env
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from transformers import AutoTokenizer  # type: ignore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import uuid


class ChromaStore:
    COLLECTION_NAME_PARENT = "volatile_storage_parent"
    COLLECTION_NAME_CHILD = "volatile_storage_child"

    def __init__(self):
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

        self.text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            chunk_size=4 * 512,  # chunk size (characters)
            chunk_overlap=0,  # chunk overlap (characters)
            tokenizer=self.main_model_tokenizer,
            add_start_index=True,  # track index in original document
        )
        self.text_splitter_small = (
            RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
                chunk_size=1 * 512,  # chunk size (characters)
                chunk_overlap=0,  # chunk overlap (characters)
                tokenizer=self.main_model_tokenizer,
                add_start_index=True,  # track index in original document
            )
        )

        self._client = chromadb.PersistentClient(path=env.get("CHROMA_PERSIST_PATH"))
        self._collection_parent = self._client.get_or_create_collection(
            self.COLLECTION_NAME_PARENT
        )
        self._collection_child = self._client.get_or_create_collection(
            self.COLLECTION_NAME_CHILD
        )
        self.vector_store_parent = Chroma(
            client=self._client,
            collection_name=self.COLLECTION_NAME_PARENT,
            embedding_function=self.embed,
        )
        self.vector_store_child = Chroma(
            client=self._client,
            collection_name=self.COLLECTION_NAME_CHILD,
            embedding_function=self.embed,
        )
        self.zone = ZoneInfo("Asia/Hong_Kong")

    def _get_now(self):
        return datetime.now().astimezone(self.zone)

    def add_documents(self, documents):
        now = self._get_now()
        now_ts = now.timestamp()
        now_str = now.strftime(env.get("DATETIME_FMT"))
        expiry_str = (now + timedelta(days=env.get("RAG_DOC_EXPIRE_DAYS"))).strftime(
            env.get("DATETIME_FMT")
        )
        for doc in documents:
            m = doc.metadata
            if m is not None:
                m["add_date"] = now_str
                m["expire_date"] = expiry_str
                m["parent_doc_id"] = ""
                m["add_ts"] = now_ts
            else:
                m = {
                    "title": "unknown",
                    "source": "unknown",
                    "add_date": now_str,
                    "expire_date": expiry_str,
                    "parent_doc_id": "",
                    "add_ts": now_ts,
                }

        parent_splits = self.text_splitter.split_documents(documents)
        ids = [uuid.uuid4().hex for _ in range(len(parent_splits))]
        for i, item in enumerate(parent_splits):
            item.metadata["parent_doc_id"] = ids[i]
        child_splits = self.text_splitter_small.split_documents(parent_splits)
        self.vector_store_parent.add_documents(parent_splits, ids=ids)
        self.vector_store_child.add_documents(child_splits)

    def get_documents(self, query, k=2):
        now = self._get_now()
        # 因为parent id可能重复，所以搜10个，后面再取k个最相关的
        small_docs, scores = zip(
            *self.vector_store_child.similarity_search_with_score(
                query,
                k=10,
                filter={
                    "add_ts": {
                        "$gt": (
                            now - timedelta(days=env.get("RAG_DOC_EXPIRE_DAYS"))
                        ).timestamp()
                    }
                }, # type: ignore
            )
        )
        min_scores: dict[str, float] = {}
        # Step 2: 遍历 small_docs 和 score，更新字典
        for doc, sc in zip(small_docs, scores):
            parent_doc_id = doc.metadata["parent_doc_id"]
            if parent_doc_id not in min_scores or sc < min_scores[parent_doc_id]:
                min_scores[parent_doc_id] = sc

        # Step 3: 从字典中提取出最小的k个 score 对应的 parent_doc_id
        sorted_min_scores = sorted(min_scores.items(), key=lambda x: x[1])
        top_two_parent_doc_ids = [item[0] for item in sorted_min_scores[:k]]
        try:
            big_docs = self.vector_store_parent.get_by_ids(top_two_parent_doc_ids)
        except Exception as e:
            print(f"Failed to retrieve documents: {e}")
            big_docs = []
        return big_docs

    def clean_old_docs(self):
        now = self._get_now()
        self._collection_child.delete(
            where={
                "add_ts": {
                    "$lt": (
                        now - timedelta(days=env.get("RAG_DOC_CLEAN_DAYS_AGO"))
                    ).timestamp()
                }
            }
        )
        self._collection_parent.delete(
            where={
                "add_ts": {
                    "$lt": (
                        now - timedelta(days=env.get("RAG_DOC_CLEAN_DAYS_AGO"))
                    ).timestamp()
                }
            }
        )


if __name__ == "__main__":
    c = ChromaStore()
    docs = c.get_documents("Paris")
    print(docs)
