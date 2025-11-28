import json
import os

def save_embedding(name, embedding, path="data/targets"):
    os.makedirs(path, exist_ok=True)
    file_path = os.path.join(path, f"{name}.json")

    with open(file_path, "w") as f:
        json.dump({"name": name, "embedding": embedding}, f)

    print(f"Embedding saved → {file_path}")
