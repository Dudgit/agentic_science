# utils/web.py
from duckduckgo_search import DDGS

def search_internet(query: str, max_results: int = 3) -> str:
    """Hits DuckDuckGo and formats the results into a readable string."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            
        if not results:
            return "No results found on the internet."
            
        formatted_results = "### WEB SEARCH RESULTS ###\n"
        for i, res in enumerate(results):
            formatted_results += f"\nResult {i+1}: {res.get('title')}\n"
            formatted_results += f"Source: {res.get('href')}\n"
            formatted_results += f"Snippet: {res.get('body')}\n"
            
        return formatted_results
        
    except Exception as e:
        return f"Internet search failed: {str(e)}"