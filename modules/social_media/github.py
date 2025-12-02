import requests
from bs4 import BeautifulSoup

def scrape_github(username):
    url = f"https://github.com/{username}"
    response = requests.get(url)

    if response.status_code != 200:
        print("GitHub profile not found")
        return None

    soup = BeautifulSoup(response.text, "html.parser")

    name = soup.find("span", class_="p-name")
    bio = soup.find("div", class_="p-note")
    followers = soup.find("span", class_="text-bold color-fg-default")
    repos = soup.find("span", class_="Counter")

    return {
        "url": url,
        "name": name.text.strip() if name else None,
        "bio": bio.text.strip() if bio else None,
        "followers": followers.text.strip() if followers else None,
        "repos": repos.text.strip() if repos else None,
    }
