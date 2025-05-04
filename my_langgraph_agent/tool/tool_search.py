from my_langgraph_agent import resource
from langchain_core.tools import tool
from typing import Annotated
from my_langgraph_agent.component.component_websearch import search as web_search

def search(query: str):
    documents = web_search(query=query)
    _ = resource.add_documents(documents)

@tool("web_search")
def web_search_returning_string(query: Annotated[str, "Keywords to search on Google."]) -> str:
    """Searches online information based on query and returns a string containing the results. If user mentions time, you MUST use other tools to get time before using this tool."""
    search(query)
    constructor = ""
    for i, doc in enumerate(resource.search_documents(query), start=1):
        constructor += """## Source ID: {}

Title: {}

URL: {}

Added Date: {}

{}

""".format(i, doc.metadata['title'], doc.metadata['score'], doc.metadata['add_date'], doc.page_content)
    # print(constructor)
    return constructor