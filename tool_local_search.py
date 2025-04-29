from globalsource import resource
from langchain_core.tools import tool
from tool_search import generate_query

@tool("search_knowledge")
def search_local_knowledge(user_question: str):
    """Search the local knowledge database. This local database contains cached search results from the past seven days. The AI should prioritize querying this database for relevant results before making requests to the online search tool."""
    query = generate_query(user_question)
    constructor = ""
    for i, doc in enumerate(resource.search_documents(query), start=1):
        constructor += """## Source ID: {}

Title: {}

URL: {}

Relevance Score: {}

Added Date: {}

{}

""".format(i, doc.metadata['title'], doc.metadata['source'], doc.metadata['score'], doc.metadata['add_date'], doc.page_content)
    return constructor