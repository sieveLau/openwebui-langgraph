from globalsource import resource
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import ValidationError
from tool_time import get_current_time
from typing import Literal

from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import add_messages
from typing import Annotated
from typing_extensions import TypedDict

from pydantic import BaseModel, Field
import datetime
llm = resource.get_main_model()

class Reflection(BaseModel):
    missing: str = Field(description="Critique of what is missing.")
    superfluous: str = Field(description="Critique of what is superfluous")


class AnswerQuestion(BaseModel):
    """Generate keywords and reflect on keywords."""

    answer: str = Field(description="~15 word keywords for google.")
    reflection: Reflection = Field(description="Your reflection on the initial keywords.")

class ResponderWithRetries:
    def __init__(self, runnable, validator):
        self.runnable = runnable
        self.validator = validator

    def respond(self, state: dict):
        response = []
        for attempt in range(3):
            response = self.runnable.invoke(
                {"messages": state["messages"]}, {"tags": [f"attempt:{attempt}"]}
            )
            try:
                self.validator.invoke(response)
                return {"messages": response}
            except ValidationError as e:
                state = state + [
                    response,
                    ToolMessage(
                        content=f"{repr(e)}\n\nPay close attention to the function schema.\n\n"
                        + self.validator.schema_json()
                        + " Respond by fixing all validation errors.",
                        tool_call_id=response.tool_calls[0]["id"],
                    ),
                ]
        return {"messages": response}

actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are expert researcher.
Current time: {time}

1. {first_instruction}
2. Reflect and critique your answer. Be severe to maximize improvement.""",
        ),
        MessagesPlaceholder(variable_name="messages"),
        (
            "user",
            "\n\n<system>Reflect on the user's original question and the"
            " actions taken thus far. Respond using the {function_name} function.</reminder>",
        ),
    ]
).partial(
    time=lambda: datetime.datetime.now().isoformat(),
)
initial_answer_chain = actor_prompt_template.partial(
    first_instruction="Provide a ~15 word keywords for google search.",
    function_name=AnswerQuestion.__name__,
) | llm.bind_tools(tools=[AnswerQuestion])
validator = PydanticToolsParser(tools=[AnswerQuestion])

first_responder = ResponderWithRetries(
    runnable=initial_answer_chain, validator=validator
)

revise_instructions = """Revise your previous answer using the new rules.
    - You should use the previous critique to add important information to your answer.
        - The keywords should be concise and relevant to the user's question.
        - The keywords should cover all the main topics mentioned in the user's question.
        - The keywords should be unique and not repetitive.
        - The keywords should be single line, and seperated by whitespace.
        - If the request contains time, a specific time in format yyyy-mm-dd is better than things like "tomorrow" or "yesterday".
    - You should use the previous critique to remove superfluous information from your answer and make SURE it is not more than 15 words.
"""


# Extend the initial answer schema to include references.
# Forcing citation in the model encourages grounded responses
class ReviseAnswer(AnswerQuestion):
    """Revise your original answer to your question. Provide an answer and reflection"""


revision_chain = actor_prompt_template.partial(
    first_instruction=revise_instructions,
    function_name=ReviseAnswer.__name__,
) | llm.bind_tools(tools=[ReviseAnswer, get_current_time])
revision_validator = PydanticToolsParser(tools=[ReviseAnswer])

revisor = ResponderWithRetries(runnable=revision_chain, validator=revision_validator)

class State(TypedDict):
    messages: Annotated[list, add_messages]


MAX_ITERATIONS = 1
builder = StateGraph(State)
builder.add_node("draft", first_responder.respond)

builder.add_node("revise", revisor.respond)
# Define looping logic:

def _get_num_iterations(state: list):
    i = 0
    for m in state[::-1]:
        if m.type not in {"tool", "ai"}:
            break
        i += 1
    return i


def event_loop(state: list):
    # in our case, we'll just stop after N plans
    num_iterations = _get_num_iterations(state["messages"])
    if num_iterations > MAX_ITERATIONS:
        return END
    return "revise"


# revise -> execute_tools OR end
builder.add_conditional_edges("revise", event_loop, ["revise", END])
builder.add_edge(START, "draft")
builder.add_edge("draft", "revise")
graph = builder.compile()

def get_query(user_question: str):
    events = graph.invoke(
        {"messages": [("user", user_question+' /no_think')]}
    )
    try:
        return events['messages'][-1].tool_calls[0]['args']['answer']
    except Exception as e:
        print(e)
        return user_question