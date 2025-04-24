from globalsource import resource
from init_env import env

from typing import Annotated
from langgraph.graph import START, StateGraph, END
from typing_extensions import List, TypedDict
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph.message import add_messages
from component_helpers import tiktoken_counter
from langchain_core.messages import AIMessage, HumanMessage, trim_messages

from tool_search import web_search_returning_string

NUM_PREDICT = 6*1024
# init api key

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
import os
llm = init_chat_model(model="deepseek-r1", model_provider="openai", base_url=env.get('BASE_URL'), max_tokens=NUM_PREDICT)

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
def generate(state: GraphState):
    from datetime import datetime
    template = f"You are an assistant for question-answering tasks. Today is {datetime.now().strftime('%Y-%m-%d, %A')}."+" Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know.\nQuestion: {question}\nContext: {context}\nAnswer:"
    prompt = ChatPromptTemplate.from_template(template)
    docs_content = web_search_returning_string(state["question"])
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
    generate
]

# Compile application and test
graph_builder = StateGraph(GraphState).add_sequence(SEQUENCE)
graph_builder.add_edge(START, "latest_user_message")
graph_builder.add_edge("generate", END)
graph = graph_builder.compile()
