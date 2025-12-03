from serpapi import GoogleSearch
from config import SERPAPI_KEY
import json

def google_search_name(query: str, num: int = 10) -> list[dict]:
    params = {
        "engine": "google",
        "q": query,
        "num": num,
        "api_key": SERPAPI_KEY,
    }

    search = GoogleSearch(params)

    try:
        results = search.get_dict()
    except json.JSONDecodeError as e:
        print("[OSINT] SerpAPI JSON decode failed:", e)
        # Optional: inspect raw response to debug key/limit issues
        try:
            raw = search.get_results()
            print("[OSINT] Raw SerpAPI response (first 300 chars):")
            print(str(raw)[:300])
        except Exception as inner:
            print("[OSINT] Also failed to read raw SerpAPI result:", inner)
        return []
    except Exception as e:
        print("[OSINT] SerpAPI request failed:", e)
        return []

    organic = results.get("organic_results", []) or []

    output = []
    for r in organic:
        output.append({
            "title": r.get("title", ""),
            "link": r.get("link", ""),
            # this helps your LinkedIn scoring (snippet-based matching)
            "snippet": r.get("snippet", ""),
        })
    return output
