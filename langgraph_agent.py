from globalsource import resource
from init_env import env

from typing import Annotated
from langgraph.graph import START, StateGraph, END
from typing_extensions import List, TypedDict
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph.message import add_messages
from component_helpers import tiktoken_counter, rules_to_string
from langchain_core.messages import AIMessage, HumanMessage, trim_messages

from datetime import timezone

from tool_search import web_search_returning_string

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
    NOW = datetime.now().astimezone(timezone.utc)
    rules = [
        f"If user's input relates to date, now it is {NOW.strftime('%Y-%m-%d %H:%M:%S UTC%z')}, a {NOW.strftime('%A')}.",
        f"If user's question contains location, you should adjust time based on their location. For example, if user is in HK, you should add 8 hours to the time given in rule 1.",
        "However, if user's question is about news in some location, you should adjust time based on that location, unless he requires you to based on his location. For example, if user asks about news in New York today and indicates his location is in HK, you should adjust time to New York's local time instead of HK's. You should judge whether the user wants you to adjust time based on their location. For example, if UTC is 2025-04-26 18:00 and user says today means HK time, although London is UTC+0 which is 2025-04-26, HK is 2025-04-27 and 04-27 is what you should use.",
        'If the question is related to date, include yyyy-mm-dd in the query.',
        "Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know."
    ]
    template = "You are an assistant for question-answering tasks. Here are the rules:\n"+rules_to_string(rules)+"\n\nQuestion: {question}\n\nContext: {context}\n\nAnswer:"
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
        max_tokens=16*1024-resource.NUM_PREDICT-50,
        start_on="human",
        end_on=("human", "tool"),
        include_system=True,
    )
    response = resource.get_main_model().invoke(full_messages)
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
