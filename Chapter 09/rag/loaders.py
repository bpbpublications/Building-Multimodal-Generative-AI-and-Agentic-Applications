import os

def load_text_documents(folder):
    print("✅ load_text_documents loaded")
    docs = {}
    for file in os.listdir(folder):
        if file.endswith(".txt"):
            with open(os.path.join(folder, file), "r", encoding="utf-8") as f:
                docs[file] = f.read()
    print("📄 Loaded text files:", list(docs.keys()))
    return docs

def load_image_paths(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
