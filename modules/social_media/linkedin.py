import time
import random
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List

from bs4 import BeautifulSoup
import requests

from modules.browser.driver import get_driver


@dataclass
class LinkedInProfile:
    """LinkedIn profile data structure"""
    url: str
    name: Optional[str] = None
    headline: Optional[str] = None
    location: Optional[str] = None
    about: Optional[str] = None
    experiences: Optional[List[Dict]] = None
    education: Optional[List[Dict]] = None
    skills: Optional[List[str]] = None


def validate_linkedin_url(url: str) -> bool:
    """
    Validate that URL is a real LinkedIn profile, not a directory page.
    
    Returns:
        True if valid profile, False otherwise
    """
    if not url:
        return False
    
    url_lower = url.lower()
    
    # Must contain linkedin.com
    if "linkedin.com" not in url_lower:
        return False
    
    # REJECT directory pages (these are useless)
    if "/pub/dir/" in url_lower:
        print(f"[LinkedIn] ✗ Rejected directory page: {url}")
        return False
    
    # REJECT search pages
    if "/search/" in url_lower or "/people/" in url_lower:
        print(f"[LinkedIn] ✗ Rejected search page: {url}")
        return False
    
    # ACCEPT individual profiles
    if "/in/" in url_lower or "/pub/" in url_lower:
        return True
    
    print(f"[LinkedIn] ✗ Unknown URL format: {url}")
    return False


def find_linkedin_profile_google(full_name: str, extra_query: str = "") -> Optional[str]:
    """
    Use Google search to find the best public LinkedIn profile URL.
    
    Args:
        full_name: Person's full name
        extra_query: Additional search terms (location, company, etc.)
    
    Returns:
        LinkedIn profile URL or None
    """
    query = f'"{full_name}" site:linkedin.com/in {extra_query}'
    url = "https://www.google.com/search"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/121.0.0.0 Safari/537.36"
        )
    }
    params = {"q": query, "num": 10}

    try:
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        if resp.status_code != 200:
            print(f"[LinkedIn] Google search failed: {resp.status_code}")
            return None

        soup = BeautifulSoup(resp.text, "html.parser")

        # Extract all links
        candidates = []
        for a in soup.select("a"):
            href = a.get("href", "")
            
            # Skip if not LinkedIn
            if "linkedin.com/in/" not in href:
                continue
            
            # Clean Google redirect URLs
            if href.startswith("/url?q="):
                href = href.split("/url?q=")[1].split("&")[0]
            
            # Validate URL
            if validate_linkedin_url(href):
                candidates.append(href)
                print(f"[LinkedIn] Found candidate: {href}")
        
        if not candidates:
            print("[LinkedIn] No valid LinkedIn profiles found in Google results")
            return None
        
        # Return first valid candidate
        return candidates[0]
    
    except Exception as e:
        print(f"[LinkedIn] Google search error: {e}")
        return None


def scrape_linkedin_profile(profile_url: str) -> Optional[Dict[str, Any]]:
    """
    Scrape a public LinkedIn profile (no login required).
    
    Args:
        profile_url: LinkedIn profile URL
    
    Returns:
        Dictionary with profile data, or None if scraping failed
    """
    # Validate URL before scraping
    if not validate_linkedin_url(profile_url):
        print(f"[LinkedIn] Invalid profile URL, skipping scrape")
        return None
    
    driver = None
    
    try:
        driver = get_driver()
        
        print(f"[LinkedIn] Opening profile: {profile_url}")
        driver.get(profile_url)
        
        # Wait for page to load
        time.sleep(random.uniform(4, 7))
        
        # Scroll to trigger lazy loading
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight * 0.3);")
        time.sleep(random.uniform(2, 3))
        
        # Get page source
        html = driver.page_source
        soup = BeautifulSoup(html, "html.parser")
        
        # Extract profile data
        profile_data = {
            "url": profile_url,
            "name": _extract_name(soup),
            "headline": _extract_headline(soup),
            "location": _extract_location(soup),
            "about": _extract_about(soup),
            "experiences": _extract_experiences(soup),
            "education": _extract_education(soup),
            "skills": _extract_skills(soup),
        }
        
        # Validate we got actual data (not a broken page)
        if not profile_data["name"] or profile_data["name"] == "Join LinkedIn":
            print("[LinkedIn] ✗ Got broken/login page instead of profile")
            return None
        
        # Print extracted data
        print("\n[LinkedIn] ✓ Successfully parsed profile:")
        print(f"  Name:      {profile_data['name']}")
        print(f"  Headline:  {profile_data['headline']}")
        print(f"  Location:  {profile_data['location']}")
        
        if profile_data['experiences']:
            print(f"  Experiences: {len(profile_data['experiences'])} found")
            for exp in profile_data['experiences'][:3]:
                print(f"    - {exp.get('title')} at {exp.get('company')}")
        
        if profile_data['education']:
            print(f"  Education: {len(profile_data['education'])} found")
        
        return profile_data
    
    except Exception as e:
        print(f"[LinkedIn] Scraping failed: {e}")
        return None
    
    finally:
        if driver:
            try:
                driver.quit()
            except:
                pass


def _extract_name(soup: BeautifulSoup) -> Optional[str]:
    """Extract profile name"""
    name_tag = soup.find("h1", class_=lambda c: c and "text-heading" in c)
    if not name_tag:
        name_tag = soup.find("h1")
    
    if name_tag:
        name = name_tag.get_text(strip=True)
        # Filter out empty or placeholder names
        if name and name != "Join LinkedIn":
            return name
    return None


def _extract_headline(soup: BeautifulSoup) -> Optional[str]:
    """Extract profile headline (job title/description)"""
    # Try multiple selectors
    selectors = [
        {"class": lambda c: c and "text-body-medium" in c},
        {"class": lambda c: c and "text-body-small" in c},
        {"class": lambda c: c and "break-words" in c},
    ]
    
    for selector in selectors:
        for div in soup.find_all("div", **selector):
            text = div.get_text(" ", strip=True)
            if text and 10 < len(text) < 200 and not text.startswith("http"):
                return text
    
    return None


def _extract_location(soup: BeautifulSoup) -> Optional[str]:
    """Extract location"""
    # Look for location indicators
    for span in soup.find_all("span", class_=lambda c: c and "text-body-small" in c):
        text = span.get_text(" ", strip=True)
        # Location usually has comma and is short
        if text and "," in text and 5 < len(text) < 80:
            # Additional validation: not a full sentence
            if not text.endswith("."):
                return text
    
    return None


def _extract_about(soup: BeautifulSoup) -> Optional[str]:
    """Extract about/summary section"""
    # Find about section
    about_section = None
    for section in soup.find_all("section"):
        aria = section.get("aria-label", "").lower()
        if "about" in aria:
            about_section = section
            break
    
    if about_section:
        # Get text, but clean it up
        about_text = about_section.get_text(" ", strip=True)
        # Remove section title
        about_text = about_text.replace("About", "", 1).strip()
        if about_text and len(about_text) > 20:
            return about_text[:1000]  # Limit length
    
    return None


def _extract_experiences(soup: BeautifulSoup) -> Optional[List[Dict]]:
    """Extract work experience"""
    # Find experience section
    exp_section = None
    for section in soup.find_all("section"):
        aria = section.get("aria-label", "").lower()
        if "experience" in aria:
            exp_section = section
            break
    
    if not exp_section:
        return None
    
    experiences = []
    
    # Find experience items (usually in list items)
    items = exp_section.find_all("li", limit=10)
    
    for li in items:
        # Try to extract job title
        title_tag = li.find("div", class_=lambda c: c and "t-bold" in c)
        if not title_tag:
            title_tag = li.find("span", class_=lambda c: c and "t-bold" in c)
        
        # Try to extract company
        company_tag = li.find("span", class_=lambda c: c and "t-normal" in c)
        
        # Try to extract date/location metadata
        meta_tag = li.find("span", class_=lambda c: c and "t-14" in c)
        if not meta_tag:
            meta_tag = li.find("span", class_=lambda c: c and "t-black--light" in c)
        
        title = title_tag.get_text(" ", strip=True) if title_tag else None
        company = company_tag.get_text(" ", strip=True) if company_tag else None
        meta = meta_tag.get_text(" ", strip=True) if meta_tag else None
        
        # Only add if we got at least a title
        if title:
            experiences.append({
                "title": title,
                "company": company,
                "duration": meta,
            })
    
    return experiences if experiences else None


def _extract_education(soup: BeautifulSoup) -> Optional[List[Dict]]:
    """Extract education history"""
    # Find education section
    edu_section = None
    for section in soup.find_all("section"):
        aria = section.get("aria-label", "").lower()
        if "education" in aria:
            edu_section = section
            break
    
    if not edu_section:
        return None
    
    education = []
    
    items = edu_section.find_all("li", limit=5)
    
    for li in items:
        school_tag = li.find("div", class_=lambda c: c and "t-bold" in c)
        if not school_tag:
            school_tag = li.find("span", class_=lambda c: c and "t-bold" in c)
        
        degree_tag = li.find("span", class_=lambda c: c and "t-14" in c)
        
        school = school_tag.get_text(" ", strip=True) if school_tag else None
        degree = degree_tag.get_text(" ", strip=True) if degree_tag else None
        
        if school:
            education.append({
                "school": school,
                "degree": degree,
            })
    
    return education if education else None


def _extract_skills(soup: BeautifulSoup) -> Optional[List[str]]:
    """Extract skills (if available on public profile)"""
    # Skills section
    skills_section = None
    for section in soup.find_all("section"):
        aria = section.get("aria-label", "").lower()
        if "skill" in aria:
            skills_section = section
            break
    
    if not skills_section:
        return None
    
    skills = []
    
    # Skills are usually in spans or divs
    for span in skills_section.find_all(["span", "div"], limit=20):
        text = span.get_text(strip=True)
        # Filter out section titles and long descriptions
        if text and 2 < len(text) < 50 and not text.lower().startswith("skill"):
            skills.append(text)
    
    # Remove duplicates while preserving order
    skills = list(dict.fromkeys(skills))
    
    return skills[:15] if skills else None  # Limit to top 15


# Legacy compatibility wrapper
def find_linkedin_profile(full_name: str, extra_query: str = "") -> Optional[str]:
    """Wrapper for backwards compatibility"""
    return find_linkedin_profile_google(full_name, extra_query)