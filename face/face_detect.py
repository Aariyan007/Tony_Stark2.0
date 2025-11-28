# pip install tensorflow-macos
from deepface import DeepFace


def detect_faces(image_path):
    result = DeepFace.extract_faces(img_path=image_path, detector_backend='retinaface')
    return result
