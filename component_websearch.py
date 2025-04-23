from langchain_core.documents import Document
import os

def search(query: str) -> list[Document]:
    """Search Google, extract HTML content using Tavily and return a list of documents."""
    PSE_ID = os.environ.get('PSE_ID', None)
    PSE_KEY = os.environ.get('PSE_KEY', None)
    TAVILY_KEY = os.environ.get('TAVILY_KEY', None)
    if not PSE_ID and PSE_KEY and TAVILY_KEY:
        print("PSE ID, PSE Key, or Tavily Key is missing.")
        raise Exception("PSE ID, PSE Key, or Tavily Key is missing.")
    
    from tavily import TavilyClient
    from googleapiclient.discovery import build
    from googleapiclient.http import HttpError

    documents: list[Document] = []
    
    ### PSE Search
    try:
        service = build("customsearch", "v1", developerKey=PSE_KEY)
    except Exception as e:
        print(f"Failed to build Google Custom Search API service. Error: {e}")
        return documents
    
    req = service.cse().list(
            q=query,
            cx=PSE_ID,
        )
    
    try:
        res = req.execute()
    except HttpError as e:
        print('Error requesting Google: {0}, reason : {1}'.format(e.status_code, e.error_details))
        return documents

    # collect first 5 urls from the search results and pass to tavily extract
    urls_title = {
        item["link"]: item["title"] for item in res.get("items", [])[:10]
    }
    client = TavilyClient(api_key=TAVILY_KEY)
    result = client.extract(urls=list(urls_title.keys()))
    if 'results' not in result:
        print('Error extracting from Tavily: ', result.get('detail', 'Unknown error'))
        return documents
    
    for item in result['results']:
        metadata = {
            "source": item.get("url", None)
        }
        metadata["title"] = urls_title.get(metadata["source"])

        document = Document(
            page_content=item.get("raw_content", "No content"),
            metadata=metadata
        )
        documents.append(document)
    return documents

if __name__ == "__main__":
    documents = search("HK weather today")
    print(documents)