from init_env import env
from globalsource import resource
from langchain_core.documents import Document
def generate_query(user_question: str) -> str:
    from datetime import datetime, timezone
    from openai import OpenAI
    from component_helpers import strip_think
    client = OpenAI(api_key=env.get('OPENAI_API_KEY'),base_url = env.get('BASE_URL'))
    NOW = datetime.now().astimezone(timezone.utc)
    rules = [
        f"If user's input relates to date, now it is {NOW.strftime('%Y-%m-%d %H:%M:%S UTC%z')}, a {NOW.strftime('%A')}.",
        f"If user's question contains location, you should adjust time based on their location. For example, if user is in HK, you should add 8 hours to the time given in rule 1.",
        "However, if user's question is about news in some location, you should adjust time based on that location, unless he requires you to based on his location. For example, if user asks about news in New York today and indicates his location is in HK, you should adjust time to New York's local time instead of HK's. You should judge whether the user wants you to adjust time based on their location. For example, if UTC is 2025-04-26 18:00 and user says today means HK time, although London is UTC+0 which is 2025-04-26, HK is 2025-04-27 and 04-27 is what you should use.",
        'If the question is related to date, include yyyy-mm-dd in the query.',
        'Your answer should be single line, concise (less than 3 phrases) and seperated by whitespace.',
        'DO NOT return double quotes.',
        'DO NOT answer the question.'
    ]
    messages = [
        {"role": "system", "content": "You are going to use Google Search. Your goal is to generate keywords to be searched by Google based on user's input. Here are the rules:\n"+'\n'.join(f'{i}. {content}' for i, content in enumerate(rules, 1))},
        {"role": "user", "content": f"{user_question}"},
    ]
    response = client.chat.completions.create(
        messages=messages,
        model="deepseek-r1",
        temperature=0.7,
        max_tokens=4096
    )
    if env.get('DEBUG'):
        print(f"------\nProcessing query: {user_question}")
        print(response.choices[0].message.content)
    resp = strip_think(response.choices[0].message.content)
    if env.get('DEBUG'):
        print("------\nGenerated query:")
        print(resp)
        print("------")
    return resp

def search(query: str):
    from component_websearch import search as web_search
    from component_helpers import embed_message, init_embed_vector_spliter
    from langchain_chroma import Chroma
    vector_store = Chroma(
        collection_name="example",
        embedding_function=resource.get_embed()
    )
    documents = web_search(query=query)
    _ = embed_message(documents, vector_store, resource.get_text_splitter())
    return vector_store

def retrieve(query: str, vector_store):
    retrieved_docs = vector_store.similarity_search(query, k=3)
    # print("------\nRetrieved documents:")
    # print(['---\nTitle: {}\nURL: {}\nRelevance: {}\n\n{}\n---'.format(doc.metadata['title'], doc.metadata['source'], doc.metadata['relevance'], doc.page_content) for doc in retrieved_docs])
    # for doc in retrieved_docs:
    #     print(f"* {doc.page_content}")
    # print("------")
    return retrieved_docs

def web_search_function(user_question: str) -> list[Document]:
    query = generate_query(user_question)
    v = search(query)
    # print("check point")
    return retrieve(query, v)
    # return []

def web_search_returning_string(user_question: str) -> str:
    query = generate_query(user_question)
    v = search(query)
    constructor = ""
    for i, doc in enumerate(retrieve(query, v), start=1):
        constructor += """## Source ID: {}

Title: {}

URL: {}

Relevance Score: {}

{}

""".format(i, doc.metadata['title'], doc.metadata['source'], doc.metadata['relevance'], doc.page_content)
    # print(constructor)
    return constructor
    

if __name__ == "__main__":
    # query = "What is the capital of France?"
    query = "What is the weather like in Paris today? I'm in New York, use my time."
    # documents = web_search_function(query)
    # print("\n\n".join(doc.page_content for doc in documents))
    # print(web_search_returning_string(query))
    print(generate_query(query))