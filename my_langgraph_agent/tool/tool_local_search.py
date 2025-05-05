from my_langgraph_agent import resource
from typing import Annotated
from langchain_core.tools import tool


@tool("search_knowledge_database")
def search_local_knowledge(
    query: Annotated[str, "Keywords to search the database"],
) -> str:
    """Search the local knowledge database. This local database contains cached online search results from the past seven days. The AI should prioritize querying this database for relevant results before making requests to the online search tool. If user mentions time, you MUST use other tools to get time before using this tool."""
    constructor = ""
    for i, doc in enumerate(resource.search_documents(query), start=1):
        constructor += """## Source ID: {}

Title: {}

URL: {}

Added Date: {}

{}

""".format(
            i,
            doc.metadata["title"],
            doc.metadata["source"],
            doc.metadata["add_date"],
            doc.page_content,
        )
    return constructor
