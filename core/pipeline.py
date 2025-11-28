from deepface import DeepFace
from core.storage import save_embedding

def process_image(image_path):
    print("Process image")

    faces = DeepFace.extract_faces(img_path=image_path, detector_backend='retinaface')
    print(f"Faces detected: {len(faces)}")

    analysis = DeepFace.analyze(
        img_path=image_path,
        actions=['age', 'gender', 'emotion', 'race'], 
        enforce_detection=False
    )

    embedding = DeepFace.represent(
        img_path=image_path,
        model_name="Facenet512",
        enforce_detection=False
    )
    
    save_embedding("user_test", embedding)

    return {"faces": faces, "analysis": analysis, "embedding": embedding}
