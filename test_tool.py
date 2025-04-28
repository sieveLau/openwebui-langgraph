from tool_search import _fake_web_search_returning_string as web_search
from globalsource import resource
from component_helpers import split_think
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import render_text_description
from langchain_core.runnables import chain
import json
import uuid
from langchain_core.tools import tool
tools = [web_search]

### convert model response to json ###
@chain
def convert_model_response_to_json(message) -> dict:
    if isinstance(message, str):
        think, json_body = split_think(message)
    else:  # Otherwise it's a chat model
        think, json_body = split_think(message.content)
    print(think)
    return json.loads(json_body[json_body.find('{'):json_body.rfind('}')+1])

############## invoking
from typing import Any, Dict, Optional, TypedDict
from langchain_core.runnables import RunnableConfig
class ToolCallRequest(TypedDict):
    """A typed dict that shows the inputs into the invoke_tool function."""

    name: str
    arguments: Dict[str, Any]

def invoke_tool(
    tool_call_request: ToolCallRequest, config: Optional[RunnableConfig] = None
):
    """A function that we can use the perform a tool invocation.

    Args:
        tool_call_request: a dict that contains the keys name and arguments.
            The name must match the name of a tool that exists.
            The arguments are the arguments to that tool.
        config: This is configuration information that LangChain uses that contains
            things like callbacks, metadata, etc.See LCEL documentation about RunnableConfig.

    Returns:
        output from the requested tool
    """
    tool_name_to_tool = {tool.name: tool for tool in tools}
    name = tool_call_request["name"]
    requested_tool = tool_name_to_tool[name]
    return requested_tool.invoke(tool_call_request["arguments"], config=config)

# @tool("tool_call_assistant")
def tool_call_proxy(user_question: str):
    """Searches the web for information."""
    rendered_tools = render_text_description(tools)
    print(rendered_tools)
    ############# test tool selection #############
    system_prompt = f"""\
You are an assistant that has access to the following set of tools. 
Here are the names and descriptions for each tool:

{rendered_tools}

Given the user input, return the name and input of the tool to use. 
Return your response as a JSON blob with 'name' and 'arguments' keys.

The `arguments` should be a dictionary, with keys corresponding 
to the argument names and the values corresponding to the requested values.
"""

    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("user", "{input}")]
    )

    chain_pre = prompt | resource.get_main_model() | convert_model_response_to_json
    tool_call_json = chain_pre.invoke({"input": user_question})
    message = invoke_tool(tool_call_json)
    id = uuid.uuid4().hex
    # return [
    #     {
    #         "role": "assistant",
    #         "content": None,
    #         "tool_calls": [
    #             {
    #                 "id": id,
    #                 "type": "function",
    #                 "function": tool_call_json
    #             }
    #         ]
    #     },
    #     {
    #         "role": "tool",
    #         "tool_call_id": id,
    #         "content": message
    #     }
    # ]
    return message