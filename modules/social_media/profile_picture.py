import requests
import re
from deepface import DeepFace
import cv2
import numpy as np

def download_image(url):
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
            return cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    except:
        return None


def extract_profile_picture(url):
    try:
        html = requests.get(url, timeout=5).text
        
        match = re.search(r"profile_pic_url\":\"(https:[^\"]+)", html)
        if match:
            img_url = match.group(1).replace("\\u0026", "&")
            return download_image(img_url)

    except:
        return None
