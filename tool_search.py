from langchain_core.documents import Document
import os
def generate_query(user_question: str) -> str:
    from datetime import datetime
    from openai import OpenAI
    import openai,os
    from component_helpers import strip_think
    # llm = init_chat_model(model="deepseek-r1", model_provider="openai", base_url=os.environ.get('BASE_URL','https://api.openai.com/v1'), max_tokens=NUM_PREDICT)
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'),base_url = os.getenv('BASE_URL','https://api.openai.com/v1'))
    messages = [
        {"role": "system", "content": "You are going to use Google Search. Generate keywords to be searched by Google based on user's input. If user's input relates to date, for your information, today is "+datetime.now().strftime("%Y-%m-%d, %A")+". If the question is related to date, include yyyy-mm-dd in the query. Your answer should be single line, concise (less than 3 phrases) and seperated by whitespace. DO NOT return double quotes. DO NOT answer the question."},
        {"role": "user", "content": f"{user_question}"},
    ]
    response = client.chat.completions.create(
        messages=messages,
        model="deepseek-r1",
        temperature=0.7,
        max_tokens=4096
    )
    print(f"------\nProcessing query: {user_question}")
    resp = strip_think(response.choices[0].message.content)
    print("------\nGenerated query:")
    print(resp)
    print("------")
    return resp

def search(query: str):
    from component_websearch import search as web_search
    from component_helpers import embed_message, init_embed_vector_spliter
    _, vector_store, text_splitter, _ = init_embed_vector_spliter(embeder_url=os.environ["EMBEDER_URL"], api_key=os.environ["OPENAI_API_KEY"])
    documents = web_search(query=query)
    _ = embed_message(documents, vector_store, text_splitter)
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
    print(constructor)
    return constructor
    

if __name__ == "__main__":
    import init_env
    query = "What is the capital of France?"
    # documents = web_search_function(query)
    # print("\n\n".join(doc.page_content for doc in documents))
    print(web_search_returning_string(query))