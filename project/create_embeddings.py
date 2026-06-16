import os
import cv2
import pickle
import numpy as np
from deepface import DeepFace

# ==============================
# CONFIGURACIÓ
# ==============================

BASE_DIR = os.path.dirname(__file__)

DATASET_PATH = os.path.join(BASE_DIR, "dataset")
OUTPUT_FILE = os.path.join(BASE_DIR, "model", "embeddings.pkl")

MODEL_NAME = "ArcFace"

# ==============================
# FUNCIONS
# ==============================

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def get_embedding(img):
    try:
        if img is None:
            return None

        # evitar cares massa petites
        if img.shape[0] < 50 or img.shape[1] < 50:
            return None

        # preprocess
        img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # embedding
        emb = DeepFace.represent(
            img_path=img,
            model_name=MODEL_NAME,
            enforce_detection=False,
            detector_backend="skip"  # 🔥 evita redetecció
        )

        emb = np.array(emb[0]["embedding"])
        return normalize(emb)

    except Exception as e:
        print("❌ Error embedding:", e)
        return None


def remove_outliers(embeddings):
    """Eliminar embeddings dolents"""
    embeddings = np.array(embeddings)

    mean = np.mean(embeddings, axis=0)

    filtered = []
    for e in embeddings:
        dist = np.linalg.norm(e - mean)
        if dist < 1.0:  # threshold
            filtered.append(e)

    return filtered


# ==============================
# MAIN
# ==============================

print("🚀 Generant embeddings...\n")

if not os.path.exists(DATASET_PATH):
    print("❌ ERROR: dataset no existeix:", DATASET_PATH)
    exit()

db = {}

for person in os.listdir(DATASET_PATH):

    person_path = os.path.join(DATASET_PATH, person)
    if not os.path.isdir(person_path):
        continue

    print(f"👤 Processant: {person}")

    embeddings = []

    for img_name in os.listdir(person_path):

        img_path = os.path.join(person_path, img_name)

        img = cv2.imread(img_path)

        if img is None:
            print(f"⚠️ No es pot llegir {img_name}")
            continue

        emb = get_embedding(img)

        if emb is not None:
            embeddings.append(emb)

    # ✅ mínim de mostres
    if len(embeddings) < 3:
        print("❌ massa pocs embeddings, ignorat\n")
        continue

    # ✅ eliminar outliers
    embeddings = remove_outliers(embeddings)

    if len(embeddings) < 3:
        print("❌ embeddings massa inconsistents\n")
        continue

    # ✅ calcular representació final
    mean_embedding = np.mean(embeddings, axis=0)
    mean_embedding = normalize(mean_embedding)

    db[person] = mean_embedding

    print(f"✅ {person}: {len(embeddings)} embeddings acceptats\n")

# ==============================
# GUARDAR
# ==============================

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

with open(OUTPUT_FILE, "wb") as f:
    pickle.dump(db, f)

print("✅ embeddings guardats a:", OUTPUT_FILE)