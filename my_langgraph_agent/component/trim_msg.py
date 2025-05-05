from typing import List

from langchain_core.messages import HumanMessage, BaseMessage, SystemMessage


def trim_messages(
    messages: List[BaseMessage], max_tokens: int, tiktoken_counter
) -> List[BaseMessage]:
    # Step 1: 识别必须保留的索引（SystemMessage 和最后一个 HumanMessage）
    must_keep_indices = list()
    last_human_index = -1

    tokens_list = []

    # 遍历消息，记录 SystemMessage 和最后一个 HumanMessage 的索引
    for i, msg in enumerate(messages):
        tokens_list.append(tiktoken_counter([msg]))
        if isinstance(msg, SystemMessage):
            must_keep_indices.append(i)
        elif isinstance(msg, HumanMessage):
            last_human_index = i

    if last_human_index != -1:
        must_keep_indices.append(last_human_index)

    must_keep_tokens = 0
    for i in must_keep_indices:
        must_keep_tokens += tokens_list[i]

    afters_index = []
    i = len(messages) - 1
    current_tokens = must_keep_tokens
    for msg in reversed(messages[last_human_index + 1 :]):
        # print(msg)
        if current_tokens + tokens_list[i] > max_tokens:
            i -= 1
            continue
        else:
            afters_index.append(i)
            current_tokens += tokens_list[i]
            i -= 1

    befores_index = []
    i = last_human_index - 1
    current_tokens = must_keep_tokens if len(afters_index) == 0 else current_tokens
    for msg in reversed(messages[:last_human_index]):
        # print(msg)
        if current_tokens + tokens_list[i] > max_tokens:
            i -= 1
            break
        else:
            befores_index.append(i)
            current_tokens += tokens_list[i]
            i -= 1
    final_indexes = sorted(must_keep_indices + afters_index + befores_index)
    final_messages = []
    for index in final_indexes:
        final_messages.append(messages[index])
    return final_messages
