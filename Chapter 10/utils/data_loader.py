import os
from PIL import Image

# ==== FILE LOADER FUNCTIONS ====
def load_texts(text_folder):
    """Load all .txt files from the folder."""
    texts = {}
    for fname in os.listdir(text_folder):
        if fname.endswith(".txt"):
            with open(os.path.join(text_folder, fname), 'r') as f:
                texts[os.path.splitext(fname)[0]] = f.read()
    return texts

def load_images(image_folder):
    """Load all .jpg files from the folder."""
    images = {}
    for fname in os.listdir(image_folder):
        if fname.endswith(".jpg"):
            path = os.path.join(image_folder, fname)
            try:
                images[os.path.splitext(fname)[0]] = Image.open(path).convert("RGB")
            except Exception as e:
                print(f"Failed to load image {fname}: {e}")
    return images

if __name__ == "__main__":
    text_data = load_texts("data/documents")
    image_data = load_images("data/images")
    print(f"Loaded {len(text_data)} text files and {len(image_data)} image files.")
