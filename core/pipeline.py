from deepface import DeepFace
from core.storage import save_embedding
from modules.social_media.google_search import google_search_name

def process_image(image_path):
    print("Process image")

    # 1. Face detection
    faces = DeepFace.extract_faces(img_path=image_path, detector_backend='retinaface', enforce_detection=False)
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

    # 4. Run OSINT search (placeholder query for now)
    search_results = google_search_name("Aariyan S Kerala computer science India LinkedIn Github Instagram Aariyan007 ")
    print("\nGoogle Search Results:")
    for r in search_results:
        print(f"- {r['title']} -> {r['link']}")

    SOCIAL_KEYWORDS = ["linkedin", "instagram", "github", "facebook", "twitter", "youtube"]

    print("\nFiltered Social Results:")
    for r in search_results:
        if any(key in r['link'].lower() for key in SOCIAL_KEYWORDS):
            print(f"{r['title']} -> {r['link']}")
    
    return {"faces": faces, "analysis": analysis, "embedding": embedding, "search": search_results}
