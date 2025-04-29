from globalsource import resource
from langchain_core.tools import tool
from tool_search import generate_query

@tool("search_knowledge")
def search_local_knowledge(user_question: str):
    """Search the local knowledge database."""
    query = generate_query(user_question)
    constructor = ""
    for i, doc in enumerate(resource.search_documents(query), start=1):
        constructor += """## Source ID: {}

Title: {}

URL: {}

{}

""".format(i, doc.metadata['title'], doc.metadata['source'], doc.page_content)
    return constructor