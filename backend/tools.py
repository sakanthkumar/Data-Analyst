from langchain_community.tools import DuckDuckGoSearchRun

def search_web(query: str):
    """
    Searches the web for the given query using DuckDuckGo.
    Returns the search summary.
    """
    try:
        search = DuckDuckGoSearchRun()
        return search.run(query)
    except Exception as e:
        return f"Web search failed: {str(e)}"
