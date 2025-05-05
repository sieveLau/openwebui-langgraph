from langchain_core.documents import Document
from my_langgraph_agent import env


def _fake_search(query: str):
    """Fake search function for testing purposes."""
    import json

    j = json.load(open("fake_search.json"))
    documents: list[Document] = []

    # collect first 5 urls from the search results and pass to tavily extract
    for item in j["results"]:
        metadata = {
            "source": item.get("url", ""),
            "relevance": item.get("score", ""),
            "title": item.get("title", ""),
        }
        content = item.get("raw_content", None)
        if content is None:
            content = item.get("content", None)
        if content is None:
            continue
        try:
            document = Document(page_content=content, metadata=metadata)
            documents.append(document)
        except Exception as e:
            print(f"Error processing document: {e}")
    return documents


def search(query: str) -> list[Document]:
    """Search the web for realtime information."""
    TAVILY_KEY = env.get("TAVILY_KEY")
    if not TAVILY_KEY:
        print("Tavily Key is missing.")
        raise Exception("Tavily Key is missing.")

    from tavily import TavilyClient

    client = TavilyClient(api_key=TAVILY_KEY)
    response = client.search(query=query, include_raw_content=True)
    documents: list[Document] = []

    # collect first 5 urls from the search results and pass to tavily extract
    for item in response["results"]:
        metadata = {
            "source": item.get("url", ""),
            "relevance": item.get("score", ""),
            "title": item.get("title", ""),
        }
        content = item.get("raw_content", None)
        if content is None:
            content = item.get("content", None)
        if content is None:
            continue
        try:
            document = Document(page_content=content, metadata=metadata)
            documents.append(document)
        except Exception as e:
            print(f"Error processing document: {e}")
    return documents


if __name__ == "__main__":
    # documents = search("HK weather today")
    documents = _fake_search("capital France")
    print(documents)
