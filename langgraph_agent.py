
from typing import Annotated
from langgraph.graph import START, StateGraph, END
from typing_extensions import List, TypedDict
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph.message import add_messages
from component_helpers import tiktoken_counter
from langchain_core.messages import AIMessage, HumanMessage, trim_messages

NUM_PREDICT = 4*1024

# init api key
import os
if not os.environ.get("OPENAI_API_KEY"):
    if os.path.exists(".env"):
        env_dict = {}
        with open(".env") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip().strip('"')
                    env_dict[key] = value
        api_key = env_dict.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set in the .env file")
        for key in env_dict:
            os.environ[key] = env_dict[key]
        os.environ["OPENAI_API_KEY"] = api_key
    else:
        raise ValueError("OPENAI_API_KEY is not set and no .env file found")

# ============= embedding BGN ================
from component_helpers import init_embed_vector_spliter
embed, vector_store, text_splitter, collection_id = init_embed_vector_spliter()
# ============= embedding END ================

# ============ Define state for application BGN ============
class GraphState(TypedDict):
    query: str
    question: str
    context: List[Document]
    messages: Annotated[list, add_messages]
    answer: str
    chroma_ids: List[str]

# FastAPI接收用的
from pydantic import BaseModel
class FastAPIState(BaseModel):
    messages: Annotated[list, add_messages]

# ============ Define state for application END ============
from langchain.chat_models import init_chat_model
llm = init_chat_model(model="deepseek-r1", model_provider="openai", base_url=os.environ.get('BASE_URL','https://api.openai.com/v1'), max_tokens=NUM_PREDICT)

from datetime import datetime
TODAY = datetime.now().strftime("%Y-%m-%d, %A")

def search(state: GraphState):
    from component_websearch import search as web_search
    from component_helpers import embed_message
    documents = web_search(query=state["query"])
    ids = embed_message(documents, vector_store, text_splitter)
    return {"chroma_ids": ids}

def clean_up(state: GraphState):
    """从 vector store 中删除本次 web search embed 的文档"""
    vector_store.delete(state["chroma_ids"])
    return {"chroma_ids": []}

from component_helpers import strip_think

def ask_ai(prompt: str, llm, feed_dict: dict) -> str:
    chain = ChatPromptTemplate.from_template(prompt) | llm | StrOutputParser()
    response = strip_think(chain.invoke(feed_dict))
    return response

def latest_user_message(state: GraphState):
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            return {"question": message.content}

def replace_last_human_message(history, new_messages):
    """
    替换 history 中最后一条 HumanMessage 为 new_messages。
    new_messages: List[BaseMessage]，通常来自 prompt.to_messages()
    """
    # 从后往前找最后一个 HumanMessage 的索引
    for i in range(len(history) - 1, -1, -1):
        if isinstance(history[i], HumanMessage):
            return history[:i] + new_messages + history[i+1:]
    # 如果没找到 HumanMessage，就直接 append（保底策略）
    return history + new_messages

# Define application steps
def generate_query(state: GraphState):
    print(f"------\nProcessing query: {state['question']}")
    resp = ask_ai("You are going to use Google Search. Generate keywords to be searched by Google based on user's input. If user's input relates to date, for your information, today is "+TODAY+". If the question is related to date, include yyyy-mm-dd in the query. Your answer should be single line, concise (less than 3 phrases) and seperated by whitespace. DO NOT return double quotes.\nUser Input: {query}\nAnswer:",
                  llm,
                  feed_dict={"query": state["question"]})
    print("------\nGenerated query:")
    print(resp)
    print("------")
    return {"query": resp}

def retrieve(state: GraphState):
    retrieved_docs = vector_store.similarity_search(state['query'], k=3)
    print("------\nRetrieved documents:")
    for doc in retrieved_docs:
        print(f"* {doc.page_content}")
    print("------")
    return {"context": retrieved_docs}

def generate(state: GraphState):
    template = f"You are an assistant for question-answering tasks. Today is {TODAY}."+" Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know.\nQuestion: {question}\nContext: {context}\nAnswer:"
    prompt = ChatPromptTemplate.from_template(template)
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    # 生成 final user message（这会返回 ChatPromptValue）
    user_prompt_value = prompt.invoke({"question": state["question"], "context": docs_content})
    # 提取其中的消息（是个 List[BaseMessage]）
    new_user_messages = user_prompt_value.to_messages()
    # 替换原始 history 中的最后一个 user message 为新生成的 prompt
    full_messages = replace_last_human_message(state["messages"], new_user_messages)
    trim_messages(
        full_messages,
        token_counter=tiktoken_counter,
        strategy="last",
        max_tokens=16*1024-NUM_PREDICT-50,
        start_on="human",
        end_on=("human", "tool"),
        include_system=True,
    )
    response = llm.invoke(full_messages)
    return {"messages": state["messages"] + [response], "answer": response.content}

SEQUENCE = [
    latest_user_message,
    generate_query,
    search,
    retrieve,
    generate,
    clean_up
]

# Compile application and test
graph_builder = StateGraph(GraphState).add_sequence(SEQUENCE)
graph_builder.add_edge(START, "latest_user_message")
graph_builder.add_edge("clean_up", END)
graph = graph_builder.compile()