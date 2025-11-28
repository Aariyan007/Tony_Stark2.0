import face_recognition

def encode_face(image):
    rgb = image[:, :, ::-1]
    return face_recognition.face_encodings(rgb)
