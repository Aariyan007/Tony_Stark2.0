from modules.browser.driver import get_driver
from deepface import DeepFace
import time
import requests
import os

def download_image(url, save_path="temp_profile.jpg"):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(save_path, "wb") as f:
            f.write(response.content)
        return save_path
    return None


def scrape_instagram(username, reference_image):
    driver = get_driver()
    url = f"https://www.instagram.com/{username}/"
    driver.get(url)
    time.sleep(5)

    try:
        # Extract profile picture
        img_url = driver.find_element("xpath", "//img[contains(@alt, 'profile picture')]").get_attribute("src")
        print("\nProfile photo:", img_url)
        
        os.makedirs("downloads", exist_ok=True)

        # Download profile image
        local_path = download_image(img_url, save_path="downloads/insta_profile.jpg")

        if not local_path:
            print("Failed to download profile picture")
            return None

        # Compare faces
        print("Comparing reference image with Instagram profile picture...")
        verify_result = DeepFace.verify(
            img1_path=reference_image,
            img2_path=local_path,
            model_name="Facenet512",
            enforce_detection=False
        )

        print("Face match result:", verify_result)

        return {
            "url": url,
            "profile_img_url": img_url,
            "match": verify_result.get("verified"),
            "distance": verify_result.get("distance")
        }

    except Exception as e:
        print("Instagram scrape failed:", e)
        return None
