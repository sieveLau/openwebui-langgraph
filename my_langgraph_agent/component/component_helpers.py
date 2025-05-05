import copy
from my_langgraph_agent import resource

from typing import List
from langchain_core.messages import convert_to_openai_messages
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from langgraph.graph.message import add_messages


def rules_to_string(rules: list[str]):
    return "\n".join(f"{i}. {content}" for i, content in enumerate(rules, 1))


def strip_think(message: str) -> str:
    if "</think>" in message:
        message = message.split("</think>")[1].lstrip("\n")
    return message


def strip_think_from_message(message: BaseMessage):
    # Must make a deepcopy here, message is the true original object
    local_message = copy.deepcopy(message)
    content: str = local_message.content  # type: ignore
    if "</think>" in content:
        index = content.find("</think>\n\n")
        messages = [content[: index + 8], content[index + 10 :]]
        local_message.content = messages[1]
    return local_message


def split_think(message: str):
    if "</think>" in message:
        index = message.find("</think>")
        messages = [message[: index + 8], message[index + 10 :]]
        return (
            messages[0]
            if messages[0].startswith("<think>")
            else "<think>\n" + messages[0]
        ), messages[1].lstrip("\n")
    else:
        return "", message


def insert_before_last_human_message(history, new_messages):
    """
    在 history 中最后一条 HumanMessage 之前插入 new_messages。
    new_messages: List[BaseMessage]，通常是 [ToolCallMessage, ToolResultMessage]
    """
    # 从后往前找最后一个 HumanMessage 的索引
    for i in range(len(history) - 1, -1, -1):
        if isinstance(history[i], HumanMessage):
            return history[:i] + new_messages + history[i:]
    # 如果没找到 HumanMessage，就直接 append（保底策略）
    return history + new_messages


def replace_last_human_message(history, new_messages):
    """
    替换 history 中最后一条 HumanMessage 为 new_messages。
    new_messages: List[BaseMessage]，通常来自 prompt.to_messages()
    """
    # 从后往前找最后一个 HumanMessage 的索引
    for i in range(len(history) - 1, -1, -1):
        if isinstance(history[i], HumanMessage):
            temp = add_messages(history[:i], new_messages)
            temp = add_messages(temp, history[i + 1 :])
            return temp
    # 如果没找到 HumanMessage，就直接 append（保底策略）
    return add_messages(history, new_messages)


def ask_ai(prompt: str, llm, feed_dict: dict) -> str:
    chain = ChatPromptTemplate.from_template(prompt) | llm | StrOutputParser()
    response = strip_think(chain.invoke(feed_dict))
    return response


# return ids of embeded documents
# you'd better save this list to remove them later
def embed_message(documents, vector_store, spliter) -> list[str]:
    all_splits = spliter.split_documents(documents)
    return vector_store.add_documents(documents=all_splits)


def tiktoken_counter(messages: List[BaseMessage]) -> int:
    """
    用 AutoTokenizer 准确计算 vllm 认为的 token 数量。For simplicity only supports str Message.contents.
    """
    tokenizer = resource.get_main_model_tokenizer()
    openai_messages = convert_to_openai_messages(messages)
    # 根据 vllm 的 /tokenize 接口测试结果，vllm是会 add_generation_prompt 的
    return len(
        tokenizer.apply_chat_template(
            convert_to_openai_messages(openai_messages),
            add_generation_prompt=True,
            tokenize=True,
        )
    )


def raw_tiktoken_counter(messages: list[dict[str, str]]) -> int:
    """
    用 AutoTokenizer 准确计算 vllm 认为的 token 数量。For simplicity only supports str Message.contents.
    """
    tokenizer = resource.get_main_model_tokenizer()

    return len(
        tokenizer.apply_chat_template(
            convert_to_openai_messages(messages),
            add_generation_prompt=True,
            tokenize=True,
        )
    )


# You can use this main to test if the result is the same as vllm's /tokenize result
if __name__ == "__main__":
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="Hello, world!"),
        AIMessage(content="Hello, how can I help you today?"),
    ]
    print(
        len(
            resource.get_main_model_tokenizer().apply_chat_template(
                convert_to_openai_messages(messages),
                add_generation_prompt=True,
                tokenize=True,
            )
        )
    )
