from serpapi import GoogleSearch
from config import SERPAPI_KEY

def google_search_name(query):
    params = {
        "engine": "google",
        "q": query,
        "api_key": SERPAPI_KEY
    }
    search = GoogleSearch(params)
    results = search.get_dict()

    output = []
    for r in results.get("organic_results", []):
        output.append({"title": r.get("title"), "link": r.get("link")})
    return output
