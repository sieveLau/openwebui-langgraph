from globalsource import resource
# from init_env import env

from typing import Annotated, TypedDict
from langgraph.graph import START, StateGraph, END
from langgraph.graph.message import add_messages
from component_helpers import tiktoken_counter
from tool_search import web_search_returning_string as web_search
from tool_local_search import search_local_knowledge
import tool_time
from langgraph.prebuilt import ToolNode
from langchain_core.messages import trim_messages

# ============ Define state for application BGN ============

# FastAPI接收用的
class State(TypedDict):
    messages: Annotated[list, add_messages]

# ============ Define state for application END ============


tools = [web_search, search_local_knowledge, tool_time.get_current_time, tool_time.convert_time]
tool_node = ToolNode(tools)
model_with_tools = resource.get_main_model().bind_tools(tools)

def should_continue(state: State):
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return END

def call_model(state: State):
    messages = state["messages"]
    messages = trim_messages(
        messages,
        token_counter=tiktoken_counter,
        strategy="last",
        max_tokens=16*1024-resource.NUM_PREDICT-50,
        start_on="human",
        include_system=True,
    )
    # print(convert_to_openai_messages(messages))
    response = model_with_tools.invoke(messages)
    return {"messages": [response]}


workflow = StateGraph(State)

# Define the two nodes we will cycle between
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue, ["tools", END])
workflow.add_edge("tools", "agent")

app = workflow.compile()