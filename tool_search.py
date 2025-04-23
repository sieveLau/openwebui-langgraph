from langchain_core.documents import Document

def generate_query(user_question: str, llm) -> str:
    from datetime import datetime
    from component_helpers import ask_ai
    print(f"------\nProcessing query: {user_question}")
    resp = ask_ai("You are going to use Google Search. Generate keywords to be searched by Google based on user's input. If user's input relates to date, for your information, today is "+datetime.now().strftime("%Y-%m-%d, %A")+". If the question is related to date, include yyyy-mm-dd in the query. Your answer should be single line, concise (less than 3 phrases) and seperated by whitespace. DO NOT return double quotes. DO NOT answer the question.\nUser Input: {query}\nAnswer:",
                  llm,
                  feed_dict={"query": user_question})
    print("------\nGenerated query:")
    print(resp)
    print("------")
    return resp

def search(query: str):
    from component_websearch import search as web_search
    from component_helpers import embed_message, init_embed_vector_spliter
    _, vector_store, text_splitter, _ = init_embed_vector_spliter(embeder_url="http://localhost:11434")
    documents = web_search(query=query)
    _ = embed_message(documents, vector_store, text_splitter)
    return vector_store

def retrieve(query: str, vector_store):
    retrieved_docs = vector_store.similarity_search(query, k=3)
    print("------\nRetrieved documents:")
    for doc in retrieved_docs:
        print(f"* {doc.page_content}")
    print("------")
    return retrieved_docs

def web_search_function(user_question: str) -> list[Document]:
    from langchain.chat_models import init_chat_model
    import os
    NUM_PREDICT = 4096
    llm = init_chat_model(model="deepseek-r1", model_provider="openai", base_url=os.environ.get('BASE_URL','https://api.openai.com/v1'), max_tokens=NUM_PREDICT)
    query = generate_query(user_question, llm)
    v = search(query)
    return retrieve(query, v)

if __name__ == "__main__":
    import init_env
    query = "What is the capital of France?"
    documents = web_search_function(query)
    print("\n\n".join(doc.page_content for doc in documents))