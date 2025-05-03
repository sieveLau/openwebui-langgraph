from globalsource import resource

from typing import Annotated, TypedDict
from langgraph.graph import START, StateGraph, END
from langgraph.graph.message import add_messages
from component_helpers import tiktoken_counter
from tool_search import web_search_returning_string as web_search
from tool_local_search import search_local_knowledge
import tool_time
from langgraph.prebuilt import ToolNode
from langchain_core.messages import trim_messages, RemoveMessage

# ============ Define state for application BGN ============


class State(TypedDict):
    messages: Annotated[list, add_messages]


# ============ Define state for application END ============

tools = [
    web_search,
    search_local_knowledge,
    tool_time.get_current_time,
    tool_time.convert_time,
]
tool_node = ToolNode(tools)
model_with_tools = resource.get_main_model().bind_tools(tools)

# TODO: replace with summary node
def trim_message_list(state: State):
    original_message_count = len(state["messages"])
    trimmed = trim_messages(
        state["messages"],
        token_counter=tiktoken_counter,
        strategy="last",
        max_tokens=16 * 1024 - resource.MIN_NUM_PREDICT,
        # this is essential, to keep at least one human message
        end_on="human",
        include_system=True,
    )
    trimmed_message_count = len(trimmed)
    remove_n = original_message_count - trimmed_message_count
    if remove_n == 0:
        return {"messages": []}
    trimmed_ids = set([message.id for message in trimmed])
    original_ids = set([message.id for message in state["messages"]])
    removed_ids = original_ids - trimmed_ids
    return {"messages": [RemoveMessage(id=id) for id in removed_ids]}


def should_continue(state: State):
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return END


def call_model(state: State):
    response = model_with_tools.invoke(state["messages"])
    return {"messages": [response]}


workflow = StateGraph(State)

# Define the two nodes we will cycle between
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)
workflow.add_node("trim", trim_message_list)

workflow.add_edge(START, "trim")
workflow.add_edge("trim", "agent")
workflow.add_conditional_edges("agent", should_continue, ["tools", END])
workflow.add_edge("tools", "trim")

app = workflow.compile()
