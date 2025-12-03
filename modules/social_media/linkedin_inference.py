# modules/social_media/linkedin_inference.py

from modules.social_media.google_search import google_search_name
from urllib.parse import urlparse
import re

def _is_linkedin_profile_url(url: str) -> bool:
    if not url:
        return False
    url = url.lower()
    return "linkedin.com" in url and ("/in/" in url or "/pub/" in url)


def _extract_username(url: str) -> str | None:
    """
    From something like:
      https://www.linkedin.com/in/aariyan007/
    return 'aariyan007'
    """
    try:
        path = urlparse(url).path  # "/in/aariyan007/"
        parts = [p for p in path.split("/") if p]
        if len(parts) >= 2 and parts[0] in ("in", "pub"):
            return parts[1].lower()
    except Exception:
        pass
    return None


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().lower()


def _score_candidate(
    result: dict,
    full_name: str,
    usernames: list[str] | None = None,
    locations: list[str] | None = None,
    keywords: list[str] | None = None,
) -> int:
    """
    Give a score 0–100 based on how well this LinkedIn result matches.
    We only use Google result metadata (title + snippet + url).
    """
    title = _normalize(result.get("title", ""))
    snippet = _normalize(result.get("snippet", ""))
    url = result.get("link", "") or result.get("url", "")
    url_lower = url.lower()

    score = 0

    # 1. Name similarity (40 pts)
    full_name_norm = _normalize(full_name)
    if full_name_norm and full_name_norm in title:
        score += 30
    elif full_name_norm and full_name_norm in snippet:
        score += 20

    # 2. LinkedIn URL structure (20 pts)
    if _is_linkedin_profile_url(url):
        score += 15

    # 3. Username pattern match (15 pts)
    if usernames:
        handle = _extract_username(url) or ""
        for u in usernames:
            u_norm = u.lower()
            if u_norm and u_norm in handle:
                score += 15
                break

    # 4. Location match (15 pts)
    if locations:
        for loc in locations:
            if _normalize(loc) in title or _normalize(loc) in snippet:
                score += 10
                break

    # 5. Keyword match (10 pts)
    if keywords:
        hit = 0
        for kw in keywords:
            if _normalize(kw) in title or _normalize(kw) in snippet:
                hit += 1
        score += min(hit * 3, 10)  # up to 10

    return min(score, 100)


def find_linkedin_profile(
    full_name: str,
    usernames: list[str] | None = None,
    locations: list[str] | None = None,
    keywords: list[str] | None = None,
) -> dict | None:
    """
    Use Google search + scoring to find the most probable LinkedIn profile.

    Returns:
      {
        "url": "...",
        "title": "...",
        "snippet": "...",
        "score": 86
      }
    or None if nothing looks legit.
    """

    # Build a fat query string
    query_parts = [full_name, "LinkedIn"]
    if locations:
        query_parts.extend(locations)
    if keywords:
        query_parts.extend(keywords)

    query = " ".join(query_parts)

    # Use your existing Google wrapper
    results = google_search_name(query)

    linkedin_candidates = []
    for r in results:
        url = r.get("link") or r.get("url", "")
        if not _is_linkedin_profile_url(url):
            continue

        score = _score_candidate(
            r,
            full_name=full_name,
            usernames=usernames,
            locations=locations,
            keywords=keywords,
        )

        r_with_score = dict(r)
        r_with_score["score"] = score
        linkedin_candidates.append(r_with_score)

    if not linkedin_candidates:
        return None

    # Sort and pick best
    linkedin_candidates.sort(key=lambda x: x["score"], reverse=True)
    best = linkedin_candidates[0]

    # If score is super low, treat as "no reliable match"
    # if best["score"] < 40:
    #     return None
    
    print("[LinkedIn] Candidates and scores:")
    for c in linkedin_candidates:
        print(f" - {c.get('link')} | score={c.get('score')} | title={c.get('title')}")


    return best
