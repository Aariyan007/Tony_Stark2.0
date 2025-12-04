# modules/social_media/instagram.py

from modules.browser.driver import get_driver
from deepface import DeepFace
import time
import requests
import os
from typing import Optional, Dict, Any


def download_image(url: str, save_path: str = "temp_profile.jpg") -> Optional[str]:
    """Download image from URL"""
    try:
        response = requests.get(url, stream=True, timeout=10)
        if response.status_code == 200:
            with open(save_path, "wb") as f:
                f.write(response.content)
            return save_path
    except Exception as e:
        print(f"[Instagram] Failed to download image: {e}")
    return None


def scrape_instagram(username: str, reference_image: str) -> Optional[Dict[str, Any]]:
    """
    Scrape Instagram profile and perform face matching.
    
    Args:
        username: Instagram username
        reference_image: Path to reference image for face matching
        
    Returns:
        Dictionary with profile data and face match results, or None if failed
    """
    driver = None
    
    try:
        driver = get_driver()
        url = f"https://www.instagram.com/{username}/"
        
        print(f"\n[Instagram] Scraping profile: {url}")
        driver.get(url)
        time.sleep(5)  # Wait for page to load
        
        # Extract profile picture URL
        try:
            img_element = driver.find_element("xpath", "//img[contains(@alt, 'profile picture')]")
            img_url = img_element.get_attribute("src")
            print(f"[Instagram] Profile photo: {img_url}")
        except Exception as e:
            print(f"[Instagram] Could not find profile picture: {e}")
            return None
        
        # Download profile image
        os.makedirs("downloads", exist_ok=True)
        local_path = download_image(img_url, save_path="downloads/insta_profile.jpg")
        
        if not local_path:
            print("[Instagram] Failed to download profile picture")
            return None
        
        # Perform face matching
        print("[Instagram] Comparing reference image with profile picture...")
        
        try:
            verify_result = DeepFace.verify(
                img1_path=reference_image,
                img2_path=local_path,
                model_name="Facenet512",
                detector_backend="retinaface",  # Use better detector
                enforce_detection=False,
                distance_metric="cosine"
            )
            
            distance = verify_result.get("distance", 1.0)
            threshold = verify_result.get("threshold", 0.4)
            verified = verify_result.get("verified", False)
            
            # Calculate confidence score (0-100)
            # Lower distance = higher confidence
            if distance <= threshold:
                confidence = max(0, min(100, (1 - distance) * 100))
            else:
                # Still give some credit if reasonably close
                confidence = max(0, (1 - distance) * 50)
            
            print(f"[Instagram] Face match result:")
            print(f"  - Verified: {verified}")
            print(f"  - Distance: {distance:.4f}")
            print(f"  - Threshold: {threshold:.4f}")
            print(f"  - Confidence: {confidence:.2f}%")
            
            return {
                "username": username,
                "url": url,
                "profile_image_url": img_url,
                "face_match_result": {
                    "verified": verified,
                    "distance": distance,
                    "threshold": threshold,
                    "confidence": confidence,
                    "model": "Facenet512",
                    "detector_backend": "retinaface",
                    "similarity_metric": "cosine",
                    "raw_result": verify_result
                }
            }
            
        except Exception as e:
            print(f"[Instagram] Face matching failed: {e}")
            return {
                "username": username,
                "url": url,
                "profile_image_url": img_url,
                "face_match_result": None
            }
    
    except Exception as e:
        print(f"[Instagram] Scraping failed: {e}")
        return None
    
    finally:
        if driver:
            try:
                driver.quit()
            except:
                pass


def scrape_instagram_enhanced(username: str, reference_image: str) -> Optional[Dict[str, Any]]:
    """
    Enhanced Instagram scraping with more profile data extraction.
    
    This version attempts to extract additional profile information like:
    - Full name
    - Bio
    - Follower count
    - Following count
    - Post count
    
    Note: Instagram's structure changes frequently, so this may need updates.
    """
    driver = None
    
    try:
        driver = get_driver()
        url = f"https://www.instagram.com/{username}/"
        
        print(f"\n[Instagram] Enhanced scraping: {url}")
        driver.get(url)
        time.sleep(5)
        
        result = {
            "username": username,
            "url": url,
            "full_name": None,
            "bio": None,
            "followers": None,
            "following": None,
            "posts": None,
            "profile_image_url": None,
            "face_match_result": None
        }
        
        # Extract profile picture
        try:
            img_element = driver.find_element("xpath", "//img[contains(@alt, 'profile picture')]")
            result["profile_image_url"] = img_element.get_attribute("src")
        except Exception as e:
            print(f"[Instagram] Could not extract profile picture: {e}")
        
        # Extract full name
        try:
            name_element = driver.find_element("xpath", "//span[@class='_ap3a _aaco _aacw _aacx _aad7 _aade']")
            result["full_name"] = name_element.text
        except Exception:
            pass
        
        # Extract bio
        try:
            bio_element = driver.find_element("xpath", "//h1[contains(@class, 'x1lliihq')]")
            result["bio"] = bio_element.text
        except Exception:
            pass
        
        # Extract stats (followers, following, posts)
        try:
            stats = driver.find_elements("xpath", "//span[@class='_ac2a']")
            if len(stats) >= 3:
                result["posts"] = int(stats[0].text.replace(',', ''))
                result["followers"] = int(stats[1].text.replace(',', ''))
                result["following"] = int(stats[2].text.replace(',', ''))
        except Exception:
            pass
        
        # Perform face matching if we have a profile image
        if result["profile_image_url"] and reference_image:
            os.makedirs("downloads", exist_ok=True)
            local_path = download_image(
                result["profile_image_url"], 
                save_path=f"downloads/insta_{username}.jpg"
            )
            
            if local_path:
                try:
                    verify_result = DeepFace.verify(
                        img1_path=reference_image,
                        img2_path=local_path,
                        model_name="Facenet512",
                        detector_backend="retinaface",
                        enforce_detection=False,
                        distance_metric="cosine"
                    )
                    
                    distance = verify_result.get("distance", 1.0)
                    threshold = verify_result.get("threshold", 0.4)
                    verified = verify_result.get("verified", False)
                    confidence = max(0, min(100, (1 - distance) * 100)) if distance <= threshold else max(0, (1 - distance) * 50)
                    
                    result["face_match_result"] = {
                        "verified": verified,
                        "distance": distance,
                        "threshold": threshold,
                        "confidence": confidence,
                        "model": "Facenet512"
                    }
                    
                    print(f"[Instagram] Face match: {confidence:.2f}% confidence")
                    
                except Exception as e:
                    print(f"[Instagram] Face matching failed: {e}")
        
        print(f"[Instagram] Extracted profile data:")
        print(f"  - Full name: {result['full_name']}")
        print(f"  - Bio: {result['bio']}")
        print(f"  - Followers: {result['followers']}")
        print(f"  - Following: {result['following']}")
        print(f"  - Posts: {result['posts']}")
        
        return result
    
    except Exception as e:
        print(f"[Instagram] Enhanced scraping failed: {e}")
        return None
    
    finally:
        if driver:
            try:
                driver.quit()
            except:
                pass


# Utility function for batch processing
def batch_scrape_instagram(usernames: list[str], reference_image: str) -> Dict[str, Any]:
    """
    Scrape multiple Instagram profiles and compare against reference image.
    
    Returns:
        Dictionary mapping username -> profile data
    """
    results = {}
    
    for username in usernames:
        print(f"\n{'='*60}")
        print(f"Processing: {username}")
        print(f"{'='*60}")
        
        result = scrape_instagram(username, reference_image)
        if result:
            results[username] = result
        
        # Be nice to Instagram's servers
        time.sleep(3)
    
    # Sort by confidence score
    sorted_results = sorted(
        results.items(),
        key=lambda x: x[1].get("face_match_result", {}).get("confidence", 0),
        reverse=True
    )
    
    print("\n" + "="*60)
    print("INSTAGRAM RESULTS SUMMARY")
    print("="*60)
    for username, data in sorted_results:
        confidence = data.get("face_match_result", {}).get("confidence", 0)
        verified = data.get("face_match_result", {}).get("verified", False)
        status = "✓" if verified else "✗"
        print(f"{status} {username}: {confidence:.2f}% confidence")
    
    return dict(sorted_results)