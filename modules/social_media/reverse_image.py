import webbrowser
import os

def open_reverse_google(image_path: str):
    """
    For now: just opens Google Lens upload page.
    You manually upload the image in the browser.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found -> {image_path}")

    print(f"Image found locally: {image_path}")
    print("Opening Google Lens in your browser...")
    print("-> In the browser, click 'Upload' and select this same image.")

    webbrowser.open("https://lens.google.com/upload")
