from deepface import DeepFace

from core.storage import save_embedding
from modules.social_media.google_search import google_search_name
from modules.social_media.username_scan import scan_usernames
from modules.social_media.profile_picture import extract_profile_picture
from modules.social_media.instagram import scrape_instagram
from modules.social_media.github import scrape_github
# from modules.social_media.linkedin import find_linkedin_profile
from modules.social_media.linkedin_inference import find_linkedin_profile as infer_linkedin_profile
from modules.social_media.linkedin import scrape_linkedin_profile






def process_image(image_path):
    """Process an image to extract face data and perform OSINT reconnaissance."""
    print("Process image")

    # ===== FACE ANALYSIS =====
    faces = _extract_face_data(image_path)
    
    # ===== OSINT SEARCH =====
    search_results = _perform_osint_search()
    
    # ===== USERNAME SCANNING =====
    usernames = _scan_social_platforms()
    
    # ===== INSTAGRAM SCRAPING =====
    instagram_result = scrape_instagram("aariyan_07", image_path)
    
    # ===== GITHUB SCRAPING =====
    github_info = scrape_github("aariyan007")
    
    # ===== LINKEDIN INFERENCE =====
    linkedin_best = _find_best_linkedin_profile()
    linkedin_profile_data = None
    if linkedin_best:
        linkedin_profile_data = scrape_linkedin_profile(linkedin_best["link"])

    
    return {
        "faces": faces["faces"],
        "analysis": faces["analysis"],
        "embedding": faces["embedding"],
        "search": search_results,
        "usernames": usernames,
        "instagram": instagram_result,
        "github": github_info,
        "linkedin": linkedin_best,
    }


def _extract_face_data(image_path):
    """Extract face detection, analysis, and embedding from image."""
    print("Extracting face data...")
    
    # 1. Face detection
    faces = DeepFace.extract_faces(
        img_path=image_path, 
        detector_backend='retinaface', 
        enforce_detection=False
    )
    print(f"Faces detected: {len(faces)}")

    # 2. Age, gender, emotion, race
    analysis = DeepFace.analyze(
        img_path=image_path,
        actions=['age', 'gender', 'emotion', 'race'], 
        enforce_detection=False
    )

    # 3. Face embedding
    embedding = DeepFace.represent(
        img_path=image_path,
        model_name="Facenet512",
        enforce_detection=False
    )
    print(f"Embedding length: {len(embedding[0]['embedding'])}")
    
    save_embedding("user_test", embedding)
    
    return {
        "faces": faces,
        "analysis": analysis,
        "embedding": embedding,
    }


def _perform_osint_search():
    """Perform OSINT search and filter social media results."""
    print("\n[OSINT] Performing search...")
    
    search_results = google_search_name("Aariyan S Kerala computer science India LinkedIn Github Instagram Aariyan007")
    print("\nGoogle Search Results:")
    for r in search_results:
        print(f"- {r['title']} -> {r['link']}")

    SOCIAL_KEYWORDS = ["linkedin", "instagram", "github", "facebook", "twitter", "youtube"]
    print("\nFiltered Social Results:")
    
    for r in search_results:
        if any(key in r['link'].lower() for key in SOCIAL_KEYWORDS):
            print(f"{r['title']} -> {r['link']}")
    
    return search_results


def _scan_social_platforms():
    """Scan usernames across social media platforms."""
    print("\nScanning usernames across platforms...")
    
    usernames = scan_usernames("aariyan")
    for u in usernames:
        print(f"[{u['platform'].upper()}] -> {u['url']}")
    
    return usernames


def _find_best_linkedin_profile():
    """Use scoring-based inference to find the most likely LinkedIn profile."""
    print("\n[OSINT] Trying to locate LinkedIn profile...")

    full_name = "Aariyan-S"

    # You can later derive these from scan_usernames(), but for now hardcode
    usernames = ["aariyan007", "aariyan_07", "aariyan","aariyan-s"]  # your known handles
    locations = ["Kerala", "India"]
    keywords = ["computer science", "developer", "CSE","ernakulam"]

    best = infer_linkedin_profile(
        full_name=full_name,
        usernames=usernames,
        locations=locations,
        keywords=keywords,
    )

    if not best:
        print("[LinkedIn] No reliable LinkedIn match found")
        return None

    print("[LinkedIn] Best candidate:")
    print(f"  URL:   {best['link']}")
    print(f"  Title: {best.get('title')}")
    print(f"  Score: {best.get('score')}")
    

    return best
