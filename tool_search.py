from init_env import env
from globalsource import resource
from langchain_core.documents import Document
from langchain_core.tools import tool
# from actor_generate_query import get_query as generate_query
def generate_query(user_question: str) -> str:
    from datetime import datetime, timezone
    client = resource.get_openai_model()
    NOW = datetime.now().astimezone(timezone.utc)
    rules = [
        f"If user's input relates to date, now it is {NOW.strftime('%Y-%m-%d %H:%M:%S UTC%z')}, a {NOW.strftime('%A')}.",
        # "If user's question contains location, you should adjust time based on their location. For example, if user is in HK, you should add 8 hours to the time given in rule 1.",
        # "However, if user's question is about news in some location, you should adjust time based on that location, unless he requires you to based on his location. For example, if user asks about news in New York today and indicates his location is in HK, you should adjust time to New York's local time instead of HK's. You should judge whether the user wants you to adjust time based on their location. For example, if UTC is 2025-04-26 18:00 and user says today means HK time, although London is UTC+0 which is 2025-04-26, HK is 2025-04-27 and 04-27 is what you should use.",
        'If the question contains time, include yyyy-mm-dd in the query.',
        'Your answer should be single line, concise (less than 3 phrases) and seperated by whitespace.',
        "Answer in English."
        'DO NOT return double quotes.',
        'DO NOT answer the question.'
    ]
    messages = [
        {"role": "system", "content": "You are going to use Google Search. Your goal is to summarize key topics in user's question. Here are the rules:\n"+'\n'.join(f'{i}. {content}' for i, content in enumerate(rules, 1))},
        {"role": "user", "content": "What's the capital of France?"},
        {"role": "assistant", "content": "Paris"},
        {"role": "user", "content": "What's the weather like today in HK?"},
        {"role": "assistant", "content": f"Weather Hong Kong {NOW.strftime('%Y-%m-%d')}"},
        {"role": "user", "content": f"{user_question}"},
    ]
    response = client.chat.completions.create(
        messages=messages,
        model=env.get('MAIN_MODEL_NAME'),
        temperature=0.7,
        max_tokens=4096
    )
    if env.get('DEBUG'):
        print(f"------\nProcessing query: {user_question}")
        print(response.choices[0].message.content)
    resp = (response.choices[0].message.content).lstrip('\n')
    if env.get('DEBUG'):
        print("------\nGenerated query:")
        print(resp)
        print("------")
    return resp

def search(query: str):
    from component_websearch import search as web_search
    documents = web_search(query=query)
    _ = resource.add_documents(documents)

def _fake_search(query: str):
    from component_websearch import _fake_search as web_search
    documents = web_search(query=query)
    _ = resource.add_documents(documents)

def web_search_function(user_question: str) -> list[Document]:
    query = generate_query(user_question)
    search(query)
    return resource.search_documents(query)

@tool("web_search")
def web_search_returning_string(question: str) -> str:
    """Searches the web for information and returns a string containing the results."""
    query = generate_query(question)
    search(query)
    constructor = ""
    for i, doc in enumerate(resource.search_documents(query), start=1):
        constructor += """## Source ID: {}

Title: {}

URL: {}

{}

""".format(i, doc.metadata['title'], doc.metadata['source'], doc.page_content)
    # print(constructor)
    return constructor

@tool("web_search")
def _fake_web_search_returning_string(question: str) -> str:
    """Searches the web for information and returns a string containing the results."""
    query = generate_query(question)
    _fake_search(query)
    constructor = ""
    for i, doc in enumerate(resource.search_documents(query), start=1):
        constructor += """## Source ID: {}

Title: {}

URL: {}

{}

""".format(i, doc.metadata['title'], doc.metadata['source'], doc.page_content)
    # print(constructor)
    return constructor

if __name__ == "__main__":
    # query = "What are the historic capitals of France?"
    query = "What's weather in HK tomorrow?"
    # query = "如何编写一个用于LLM Agent的def get_current_time(utc_offset: int) -> str工具？"
    # documents = web_search_function(query)
    # print("\n\n".join(doc.page_content for doc in documents))
    # print(web_search_returning_string(query))
    # print(_fake_web_search_returning_string.name)
    # print(_fake_web_search_returning_string.description)
    # print(_fake_web_search_returning_string.args)
    # print(generate_query(query))
    result = generate_query('weather in New York on 2024-07-08')
    print(result)