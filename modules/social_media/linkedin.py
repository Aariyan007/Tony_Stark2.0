import time
import random
from dataclasses import dataclass, asdict

from bs4 import BeautifulSoup
import requests

from modules.browser.driver import get_driver


@dataclass
class LinkedInProfile:
    url: str
    name: str | None = None
    headline: str | None = None
    location: str | None = None
    about: str | None = None
    experiences: list | None = None


# ---------- 1) Find LinkedIn profile via Google ----------

def find_linkedin_profile(full_name: str, extra_query: str = "") -> str | None:
    """
    Use Google search to find the best public LinkedIn profile URL
    for this name. No API keys, just simple HTML search.
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
    params = {"q": query, "num": 5}

    resp = requests.get(url, headers=headers, params=params, timeout=10)
    if resp.status_code != 200:
        print("[LinkedIn] Google search failed:", resp.status_code)
        return None

    soup = BeautifulSoup(resp.text, "html.parser")

    for a in soup.select("a"):
        href = a.get("href", "")
        if "linkedin.com/in/" in href:
            # Google links are like "/url?q=REAL_URL&sa=..."
            if href.startswith("/url?q="):
                href = href.split("/url?q=")[1].split("&")[0]
            print("[LinkedIn] Found candidate profile:", href)
            return href

    print("[LinkedIn] No LinkedIn profile found in Google results")
    return None


# ---------- 2) Scrape a LinkedIn profile page (public) ----------

def scrape_linkedin_profile(profile_url: str) -> dict | None:
    """
    Scrape a public LinkedIn profile (no login).
    This will break if LinkedIn changes their HTML, but it's good enough
    for a personal OSINT project.
    """
    driver = get_driver()

    print(f"[LinkedIn] Opening profile: {profile_url}")
    driver.get(profile_url)

    # Let content load + lazy sections render
    time.sleep(random.uniform(4, 7))

    # Scroll a bit to force more content
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight * 0.3);")
    time.sleep(random.uniform(2, 3))

    html = driver.page_source
    soup = BeautifulSoup(html, "html.parser")

    # --- name ---
    name = None
    name_tag = soup.find("h1")
    if name_tag:
        name = name_tag.get_text(strip=True)

    # --- headline ---
    headline = None
    # LinkedIn often uses <div> with text-body-medium or text-body-small under the hero section
    for div in soup.find_all("div"):
        cls = " ".join(div.get("class", []))
        if "text-body-medium" in cls or "text-body-small" in cls:
            text = div.get_text(" ", strip=True)
            if text and len(text) < 200:
                headline = text
                break

    # --- location ---
    location = None
    # usually small text right under name/headline
    for span in soup.find_all("span"):
        text = span.get_text(" ", strip=True)
        if text and "," in text and len(text) < 80:
            # very crude heuristic, but works okay for cities/regions
            location = text
            break

    # --- about section ---
    about = None
    about_section = None
    for section in soup.find_all("section"):
        aria = section.get("aria-label", "").lower()
        if "about" in aria:
            about_section = section
            break
    if about_section:
        about = about_section.get_text(" ", strip=True)

    # --- experience section ---
    experiences = []
    exp_section = None
    for section in soup.find_all("section"):
        aria = section.get("aria-label", "").lower()
        if "experience" in aria:
            exp_section = section
            break

    if exp_section:
        # grab first few roles
        items = exp_section.find_all("li")[:5]
        for li in items:
            title = li.find("div", {"class": lambda c: c and "t-bold" in c})  # job title-ish
            company = li.find("span", {"class": lambda c: c and "t-normal" in c})
            date_loc = li.find("span", {"class": lambda c: c and "t-14" in c})

            experiences.append({
                "title": title.get_text(" ", strip=True) if title else None,
                "company": company.get_text(" ", strip=True) if company else None,
                "meta": date_loc.get_text(" ", strip=True) if date_loc else None,
            })

    profile = LinkedInProfile(
        url=profile_url,
        name=name,
        headline=headline,
        location=location,
        about=about,
        experiences=experiences or None,
    )

    print("\n[LinkedIn] Parsed profile:")
    print(f"  Name:      {profile.name}")
    print(f"  Headline:  {profile.headline}")
    print(f"  Location:  {profile.location}")
    if profile.experiences:
        print("  Experiences:")
        for exp in profile.experiences:
            print("   -", exp)

    return asdict(profile)
