from globalsource import resource
from init_env import env

from typing import Annotated
from langgraph.graph import START, StateGraph, END
from typing_extensions import List, TypedDict
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph.message import add_messages
from component_helpers import tiktoken_counter, rules_to_string, split_think, insert_before_last_human_message, replace_last_human_message, ensure_system_message
from langchain_core.messages import BaseMessage, HumanMessage, trim_messages
from test_tool import tool_call_proxy
from datetime import datetime,timezone

# init api key

# ============ Define state for application BGN ============
class GraphState(TypedDict):
    query: str
    question: str
    context: List[str]
    messages: List[BaseMessage]
    answer: str
    chroma_ids: List[str]

# FastAPI接收用的
from pydantic import BaseModel
class FastAPIState(BaseModel):
    messages: Annotated[list, add_messages]

# ============ Define state for application END ============
rules = [
    "If user's input relates to date, now it is {}, a {}.",
    "If user's question contains location, you should adjust time based on their location. For example, if user is in HK, you should add 8 hours to the time given in rule 1.",
    "However, if user's question is about news in some location, you should adjust time based on that location, unless he requires you to based on his location. For example, if user asks about news in New York today and indicates his location is in HK, you should adjust time to New York's local time instead of HK's. You should judge whether the user wants you to adjust time based on their location. For example, if UTC is 2025-04-26 18:00 and user says today means HK time, although London is UTC+0 which is 2025-04-26, HK is 2025-04-27 and 04-27 is what you should use.",
    'If the question is related to date, include yyyy-mm-dd in the query.',
    "If you don't know the answer, just say that you don't know."
]
def check_system_message(state: GraphState):
    NOW = datetime.now().astimezone(timezone.utc)
    rules[0]=rules[0].format(NOW.strftime('%Y-%m-%d %H:%M:%S UTC%z'), NOW.strftime('%A'))
    modified = ensure_system_message(state["messages"], rules, "You are an helpful assistant. Here are the rules:\n", NOW)
    return {"messages": modified}

def latest_user_message(state: GraphState):
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            return {"question": message.content}

# the parameter is passed by value
def prepare_context(state: GraphState):
    rules = [
        "DO NOT answer user's question by yourself. Your job is to determine whether to call your assistant.",
        "Answer YES if you need the assistant to collect information to you. Otherwise, answer NO.",
        "If history conversation contains enough information for you to answer user's question, you don't have to ask the assistant."
    ]
    template = "You are a helpful assistant preparing information for a senior bot. You have another assistant to help you if you need external information. You are determining whether to call the assistant to answer user's question. Here are the rules:\n"+rules_to_string(rules)+"\n\nUser's question: {question}\n\nExisting context: {context}\n\nCall your assistant?"
    prompt = ChatPromptTemplate.from_template(template)

    USER_QUESTION = state["question"]
    MESSAGES = state["messages"]
    
    tries = 0
    while True and tries < 2:
        current_context = state['context']
        user_prompt_value = prompt.invoke({"question": USER_QUESTION, "context": '\n\n'.join(current_context)}).to_messages()
        full_messages = replace_last_human_message(MESSAGES, user_prompt_value)
        full_messages = trim_messages(
            full_messages,
            token_counter=tiktoken_counter,
            strategy="last",
            max_tokens=16*1024-resource.NUM_PREDICT-50,
            start_on="human",
            end_on=("human", "tool"),
            include_system=True,
        )
        response = resource.get_main_model().invoke(full_messages)
        answer = split_think(response.content)
        print(answer[0])
        if answer[1].lower() == "yes":
            state['context'].append(tool_call_proxy(USER_QUESTION))
            tries += 1
        else: 
            break
    return {"context": state['context']}

# Define application steps
def generate(state: GraphState):

    template = "Context: {context}\n\n{question}"
    prompt = ChatPromptTemplate.from_template(template)
    user_prompt_value = prompt.invoke({"question": state['question'], "context": '\n\n'.join(state['context'])}).to_messages()
    full_messages = replace_last_human_message(state["messages"], user_prompt_value)

    ## 恢复到把 内容 和 context merge 到一起的方案
    full_messages = trim_messages(
        full_messages,
        token_counter=tiktoken_counter,
        strategy="last",
        max_tokens=16*1024-resource.NUM_PREDICT-50,
        start_on="human",
        end_on=("human", "tool"),
        include_system=True,
    )

    response = resource.get_main_model().invoke(full_messages)
    return {"messages": add_messages(state["messages"], [response]), "answer": response.content}

SEQUENCE = [
    check_system_message,
    latest_user_message,
    prepare_context,
    generate
]

# Compile application and test
graph_builder = StateGraph(GraphState).add_sequence(SEQUENCE)
graph_builder.add_edge(START, "check_system_message")
# graph_builder.add_edge("generate", END)
graph = graph_builder.compile()