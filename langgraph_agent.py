from globalsource import resource

from typing import Annotated, TypedDict
from langgraph.graph import START, StateGraph, END
from langgraph.graph.message import add_messages
from component_helpers import tiktoken_counter
from tool_search import web_search_returning_string as web_search
from tool_local_search import search_local_knowledge
import tool_time
from langgraph.prebuilt import ToolNode
from langchain_core.messages import RemoveMessage, ToolCall
from trim_msg import trim_messages

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
    messages = state["messages"]
    trimmed = trim_messages(messages, 16 * 1024 - resource.MIN_NUM_PREDICT, tiktoken_counter)
    trimmed_ids = set([message.id for message in trimmed])
    original_ids = set([message.id for message in messages])
    removed_ids = original_ids - trimmed_ids
    return {"messages": [RemoveMessage(id=id) for id in removed_ids]}

def fixing(state: State):
    messages = state["messages"]
    last_message = messages[-1]
    invalid_call = last_message.invalid_tool_calls[0]
    
    
    new_tool_call = ToolCall(
        name=invalid_call['name'],
        args={},
        id=invalid_call['id']
    )
    last_message.tool_calls=new_tool_call,
    last_message.invalid_tool_calls=[]
    print(last_message)
    return {"messages": [last_message]}

def should_continue(state: State):
    messages = state["messages"]
    last_message = messages[-1]
    print(last_message)
    if last_message.tool_calls:
        return "tools"
    ## Reroute to a fixing node to manipulate the record to create a valid tool call.
    ## TODO: detect true error
    elif len(last_message.invalid_tool_calls) > 0:
        return "fixing"
    return END


def call_model(state: State):
    response = model_with_tools.invoke(state["messages"])
    return {"messages": [response]}


workflow = StateGraph(State)

# Define the two nodes we will cycle between
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)
workflow.add_node("trim", trim_message_list)
workflow.add_node("fixing", fixing)

workflow.add_edge(START, "trim")
workflow.add_edge("trim", "agent")
workflow.add_conditional_edges("agent", 
                               # Next, we pass in the function that will determine which node is called next.
                               should_continue, 
                               # Next, we pass in the path map - all the possible nodes this edge could go to
                               ["tools", "fixing", END])
workflow.add_edge("tools", "trim")
workflow.add_edge("fixing", "tools")

app = workflow.compile()
