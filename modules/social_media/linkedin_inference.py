# modules/social_media/linkedin_inference.py

from modules.social_media.google_search import google_search_name
from urllib.parse import urlparse
import re
from typing import Optional, List, Dict, Any


def _is_linkedin_profile_url(url: str) -> bool:
    """Check if URL is a valid LinkedIn profile (not a directory page)"""
    if not url:
        return False
    
    url = url.lower()
    
    # Must contain linkedin.com
    if "linkedin.com" not in url:
        return False
    
    # REJECT directory pages (these are useless)
    if "/pub/dir/" in url:
        return False
    
    # REJECT search pages
    if "/search/" in url or "/people/" in url:
        return False
    
    # ACCEPT individual profiles
    if "/in/" in url or "/pub/" in url:
        return True
    
    return False


def _extract_username(url: str) -> Optional[str]:
    """
    Extract username from LinkedIn URL.
    
    Examples:
      https://www.linkedin.com/in/aariyan007/ -> 'aariyan007'
      https://in.linkedin.com/in/aariyan-s-123456 -> 'aariyan-s-123456'
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
    """Normalize text for comparison"""
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip().lower()


def _score_candidate(
    result: Dict[str, Any],
    full_name: str,
    usernames: Optional[List[str]] = None,
    locations: Optional[List[str]] = None,
    keywords: Optional[List[str]] = None,
) -> int:
    """
    Score a LinkedIn search result based on how well it matches the target person.
    
    Scoring breakdown:
    - Name match: 40 points
    - Valid profile URL: 20 points
    - Username match: 15 points
    - Location match: 15 points
    - Keyword match: 10 points
    
    Returns score 0-100
    """
    title = _normalize(result.get("title", ""))
    snippet = _normalize(result.get("snippet", ""))
    url = result.get("link", "") or result.get("url", "")
    url_lower = url.lower()
    
    score = 0
    
    # 1. NAME SIMILARITY (40 points max)
    full_name_norm = _normalize(full_name)
    
    if full_name_norm:
        # Exact match in title
        if full_name_norm in title:
            score += 40
        # Exact match in snippet
        elif full_name_norm in snippet:
            score += 30
        # Partial match (first name or last name)
        else:
            name_parts = full_name_norm.split()
            for part in name_parts:
                if len(part) >= 3 and part in title:
                    score += 15
                    break
    
    # 2. VALID LINKEDIN PROFILE URL (20 points)
    if _is_linkedin_profile_url(url):
        score += 20
    else:
        # Heavily penalize non-profile pages
        score -= 50
        return max(0, score)  # Early return if not a valid profile
    
    # 3. USERNAME PATTERN MATCH (15 points)
    if usernames:
        handle = _extract_username(url) or ""
        for u in usernames:
            u_norm = _normalize(u)
            if u_norm and u_norm in handle:
                score += 15
                break
            # Partial username match
            elif u_norm and len(u_norm) >= 4:
                # Check if username is substring of handle
                if u_norm in handle:
                    score += 10
                    break
    
    # 4. LOCATION MATCH (15 points)
    if locations:
        location_hits = 0
        for loc in locations:
            loc_norm = _normalize(loc)
            if loc_norm in title or loc_norm in snippet:
                location_hits += 1
        
        # Scale points based on number of location matches
        if location_hits > 0:
            score += min(location_hits * 8, 15)
    
    # 5. KEYWORD MATCH (10 points)
    if keywords:
        keyword_hits = 0
        for kw in keywords:
            kw_norm = _normalize(kw)
            if kw_norm and (kw_norm in title or kw_norm in snippet):
                keyword_hits += 1
        
        # Scale points based on number of keyword matches
        if keyword_hits > 0:
            score += min(keyword_hits * 3, 10)
    
    return min(score, 100)


def find_linkedin_profile(
    full_name: str,
    usernames: Optional[List[str]] = None,
    locations: Optional[List[str]] = None,
    keywords: Optional[List[str]] = None,
    min_score: int = 40,
) -> Optional[Dict[str, Any]]:
    """
    Find the most likely LinkedIn profile using Google search and scoring.
    
    Args:
        full_name: Person's full name
        usernames: List of known usernames/handles
        locations: List of locations associated with the person
        keywords: List of keywords (job titles, skills, schools, etc.)
        min_score: Minimum score required to return a result (default: 40)
    
    Returns:
        Best matching LinkedIn profile with score, or None if no good match
        {
            "link": "https://linkedin.com/in/username",
            "title": "Person Name - Job Title",
            "snippet": "Profile description...",
            "score": 85
        }
    """
    # Build search query
    query_parts = [f'"{full_name}"', "LinkedIn"]
    
    # Add locations (top 2)
    if locations:
        query_parts.extend(locations[:2])
    
    # Add keywords (top 3)
    if keywords:
        query_parts.extend(keywords[:3])
    
    query = " ".join(query_parts)
    
    print(f"[LinkedIn] Search query: {query}")
    
    # Perform Google search
    try:
        results = google_search_name(query)
    except Exception as e:
        print(f"[LinkedIn] Google search failed: {e}")
        return None
    
    # Filter and score candidates
    linkedin_candidates = []
    
    for r in results:
        url = r.get("link") or r.get("url", "")
        
        # Skip if not a LinkedIn URL
        if "linkedin.com" not in url.lower():
            continue
        
        # Skip directory pages and search pages
        if not _is_linkedin_profile_url(url):
            print(f"[LinkedIn] Skipping non-profile URL: {url}")
            continue
        
        # Score this candidate
        score = _score_candidate(
            r,
            full_name=full_name,
            usernames=usernames,
            locations=locations,
            keywords=keywords,
        )
        
        # Create scored result
        r_with_score = {
            "link": url,
            "title": r.get("title", ""),
            "snippet": r.get("snippet", ""),
            "score": score,
        }
        
        linkedin_candidates.append(r_with_score)
    
    if not linkedin_candidates:
        print("[LinkedIn] No valid LinkedIn profiles found in search results")
        return None
    
    # Sort by score (descending)
    linkedin_candidates.sort(key=lambda x: x["score"], reverse=True)
    
    # Print all candidates for debugging
    print("\n[LinkedIn] Candidates and scores:")
    for c in linkedin_candidates[:5]:  # Show top 5
        print(f"  {c['score']:3d} | {c['link']}")
        print(f"      Title: {c['title'][:80]}")
    
    # Get best candidate
    best = linkedin_candidates[0]
    
    # Check if score meets minimum threshold
    if best["score"] < min_score:
        print(f"\n[LinkedIn] Best score ({best['score']}) below threshold ({min_score})")
        print(f"[LinkedIn] No reliable match found")
        return None
    
    print(f"\n[LinkedIn] ✓ Best candidate:")
    print(f"  URL:   {best['link']}")
    print(f"  Title: {best['title']}")
    print(f"  Score: {best['score']}")
    
    return best


def find_multiple_linkedin_profiles(
    full_name: str,
    usernames: Optional[List[str]] = None,
    locations: Optional[List[str]] = None,
    keywords: Optional[List[str]] = None,
    min_score: int = 40,
    max_results: int = 3,
) -> List[Dict[str, Any]]:
    """
    Find multiple potential LinkedIn profiles (for ambiguous cases).
    
    Returns:
        List of candidate profiles sorted by score
    """
    # Build search query
    query = f'"{full_name}" LinkedIn'
    if locations:
        query += f" {locations[0]}"
    
    print(f"[LinkedIn] Multi-search query: {query}")
    
    try:
        results = google_search_name(query)
    except Exception as e:
        print(f"[LinkedIn] Search failed: {e}")
        return []
    
    candidates = []
    
    for r in results:
        url = r.get("link", "")
        
        if not _is_linkedin_profile_url(url):
            continue
        
        score = _score_candidate(r, full_name, usernames, locations, keywords)
        
        if score >= min_score:
            candidates.append({
                "link": url,
                "title": r.get("title", ""),
                "snippet": r.get("snippet", ""),
                "score": score,
            })
    
    # Sort and limit results
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:max_results]


# Example usage
if __name__ == "__main__":
    # Test with sample data
    result = find_linkedin_profile(
        full_name="Aariyan S",
        usernames=["aariyan007", "aariyan-s", "aariyan_07"],
        locations=["Kerala", "India", "Ernakulam"],
        keywords=[
            "computer science",
            "developer",
            "CSE",
            "Muthoot Institute of Technology and Science"
        ],
        min_score=40
    )
    
    if result:
        print(f"\n✓ Found profile: {result['link']}")
        print(f"  Score: {result['score']}")
    else:
        print("\n✗ No reliable LinkedIn profile found")