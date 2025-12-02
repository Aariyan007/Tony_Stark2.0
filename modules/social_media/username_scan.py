import requests

SOCIAL_PLATFORMS = {
    "instagram": "https://www.instagram.com/{}",
    "github": "https://github.com/{}",
    "twitter": "https://www.twitter.com/{}",
    "facebook": "https://www.facebook.com/{}",
    "youtube": "https://www.youtube.com/@{}",
    "reddit": "https://www.reddit.com/user/{}"
}

def check_username(username):
    results = []
    for platform, url_template in SOCIAL_PLATFORMS.items():
        url = url_template.format(username)
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                results.append({"platform": platform, "username": username, "url": url})
        except:
            pass
    return results


def scan_usernames(base_name):
    variations = [
        base_name,
        base_name + "07",
        base_name + "007",
        base_name + "_07",
        base_name + "_official",
        base_name + "_cs"
    ]

    found = []
    for user in variations:
        found.extend(check_username(user))

    return found
