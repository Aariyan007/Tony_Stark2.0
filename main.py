from core.pipeline import process_image
# from pprint import pprint

result = process_image("test.jpg")
print("Embedding shape:", len(result["embedding"][0]["embedding"]))

