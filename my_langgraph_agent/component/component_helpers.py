from my_langgraph_agent.globalsource import resource

from langchain_core.messages import BaseMessage
from typing import List
from langchain_core.messages import convert_to_openai_messages
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph.message import add_messages
from datetime import datetime, timezone


def rules_to_string(rules: list[str]):
    return '\n'.join(f'{i}. {content}' for i, content in enumerate(rules, 1))

def strip_think(message: str) -> str:
    if "</think>" in message:
        message = message.split("</think>")[1].lstrip('\n')
    return message

def split_think(message: str):
    if "</think>" in message:
        index = message.find("</think>")
        messages = [
            message[:index+8],
            message[index+9:]
        ]
        return (messages[0] if messages[0].startswith('<think>') else '<think>\n' + messages[0]), messages[1].lstrip('\n')
    else:
        return '', message
    
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
            temp = add_messages(temp, history[i+1:])
            return temp
    # 如果没找到 HumanMessage，就直接 append（保底策略）
    return add_messages(history, new_messages)

def ensure_system_message(history: List[BaseMessage], rules: List[str], template_prefix: str, NOW) -> List[BaseMessage]:
    """
    确保 history 中存在符合要求的 SystemMessage。
    - 如果没有 SystemMessage，则在开头插入一条。
    - 如果已有 SystemMessage，但时间过期了，则更新。
    rules: 当前的规则列表
    template_prefix: 开头的字符串，比如 "You are a helpful assistant. Here are the rules:\n"
    """
    expected_rule_0_prefix = f"If user's input relates to date, now it is {NOW.strftime('%Y-%m-%d %H')}"

    # 组装新的SystemMessage内容
    def build_system_message():
        rules_updated = rules.copy()
        rules_updated[0] = f"If user's input relates to date, now it is {NOW.strftime('%Y-%m-%d %H:%M:%S UTC%z')}, a {NOW.strftime('%A')}."
        return [SystemMessage(content=template_prefix + rules_to_string(rules_updated))]

    for i, message in enumerate(history):
        if isinstance(message, SystemMessage):
            # 检查是否需要更新
            if not message.content.startswith(template_prefix):
                # 不是我们的system，跳过不管
                continue
            if expected_rule_0_prefix not in message.content:
                # 说明日期旧了，需要替换
                history[i] = build_system_message()
            return history  # 找到并处理了，直接返回

    # 如果没有找到SystemMessage，插入
    return add_messages(build_system_message(), history)


def ask_ai(prompt: str, llm, feed_dict: dict) -> str:
    chain = ChatPromptTemplate.from_template(prompt) | llm | StrOutputParser()
    response = strip_think(chain.invoke(feed_dict))
    return response

# return ids of embeded documents
# you'd better save this list to remove them later
def embed_message(documents, vector_store, spliter) -> list[str]:
    all_splits = spliter.split_documents(documents)
    return vector_store.add_documents(documents=all_splits)

def init_embed_vector_spliter(embeder_url, api_key, persist_directory = None, seperated=True, separators='\n', chunk_size=1000, chunk_overlap=0, add_start_index=True):
    # from langchain_text_splitters import CharacterTextSplitter
    # from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_chroma import Chroma
    import uuid
    embed = resource.get_embed()
    collection_id = uuid.uuid4().hex if seperated else "global"
    vector_store = Chroma(
        collection_name=collection_id,
        embedding_function=embed,
        persist_directory=persist_directory
    )
    return embed, vector_store, resource.get_text_splitter(), collection_id

def tiktoken_counter(messages: List[BaseMessage]) -> int:
    """
    用 AutoTokenizer 准确计算 vllm 认为的 token 数量。For simplicity only supports str Message.contents.
    """
    tokenizer = resource.get_main_model_tokenizer()
    openai_messages = convert_to_openai_messages(messages)
    # 根据 vllm 的 /tokenize 接口测试结果，vllm是会 add_generation_prompt 的
    return len(tokenizer.apply_chat_template(convert_to_openai_messages(openai_messages), add_generation_prompt=True, tokenize=True))

def raw_tiktoken_counter(messages: list[dict[str: str]]) -> int:
    """
    用 AutoTokenizer 准确计算 vllm 认为的 token 数量。For simplicity only supports str Message.contents.
    """
    tokenizer = resource.get_main_model_tokenizer()

    return len(tokenizer.apply_chat_template(convert_to_openai_messages(messages), add_generation_prompt=True, tokenize=True))

# You can use this main to test if the result is the same as vllm's /tokenize result
if __name__ == "__main__":
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="Hello, world!"),
        AIMessage(content="Hello, how can I help you today?")
    ]
    print(len(resource.get_main_model_tokenizer().apply_chat_template(convert_to_openai_messages(messages), add_generation_prompt=True, tokenize=True)))