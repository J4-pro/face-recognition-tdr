import cv2
import os
import time
import numpy as np
import pickle

from flask import Flask, Response, render_template, jsonify
from ultralytics import YOLO
from multiprocessing import Process, Queue, Manager, freeze_support
from deepface import DeepFace
from collections import Counter
from persones import persones

# ---------------- CONFIG ----------------
FPS = 30
FRAME_TIME = 1.0 / FPS
FRAME_WIDTH = 640

THRESHOLD = 0.45
EMBEDDINGS_FILE = "models/embeddings.pkl"

app = Flask(__name__)

model = YOLO("models/yolov8n-face.pt")
camera = cv2.VideoCapture(1)

face_queue = Queue(maxsize=5)

tracked_faces = None
manager = None
face_id_counter = 0

# ---------------- EMBEDDING ----------------
def get_embedding(img):
    try:
        emb = DeepFace.represent(
            img_path=img,
            model_name="ArcFace",
            enforce_detection=False
        )
        return np.array(emb[0]["embedding"])
    except:
        return None

def cosine_distance(a, b):
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ---------------- LOAD EMBEDDINGS ----------------
def carregar_base_dades():
    if not os.path.exists(EMBEDDINGS_FILE):
        print("❌ No existeix embeddings.pkl. Executa build_embeddings.py primer")
        exit()

    with open(EMBEDDINGS_FILE, "rb") as f:
        db = pickle.load(f)

    print("✅ Embeddings carregats:", db.keys())
    return db

# ---------------- WORKER ----------------
def worker_reconeixement(queue, tracked_faces, db_embeddings):
    while True:
        data = queue.get()

        if data is None:
            break

        frame, x1, y1, x2, y2, fid = data

        # padding
        pad = 20
        h, w = frame.shape[:2]

        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)

        face_img = frame[y1:y2, x1:x2]

        if face_img.size == 0:
            continue

        emb = get_embedding(face_img)
        if emb is None:
            continue

        millor_nom = "unknown"
        millor_dist = 999

        # comparar embeddings
        for persona, emb_list in db_embeddings.items():
            for emb_db in emb_list:
                dist = cosine_distance(emb, emb_db)

                if dist < millor_dist:
                    millor_dist = dist
                    millor_nom = persona

        # threshold
        if millor_dist > 0.6:
            millor_nom = "unknown"
        elif millor_dist > 0.5:
            millor_nom = "dubtós"

        print(f"[PRO] {fid}: {millor_nom} ({millor_dist})")

        # estabilitat amb historial
        if fid in tracked_faces:
            hist = tracked_faces[fid].get("history", [])
            hist.append(millor_nom)

            if len(hist) > 5:
                hist.pop(0)

            noms_valids = [n for n in hist if n != "unknown"]

            if noms_valids:
                final = Counter(noms_valids).most_common(1)[0][0]
            else:
                final = "unknown"

            info = persones.get(final, {})

            tracked_faces[fid].update({
                "nom": final,
                "distancia": millor_dist,
                "success": final != "unknown",
                "history": hist,
                "edat": info.get("edat"),
                "rol": info.get("rol"),
                "img": info.get("imatge_ref")
            })

# ---------------- TRACKING ----------------
def get_face_id(x1, y1, x2, y2):
    global face_id_counter, tracked_faces, manager

    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    for fid, data in tracked_faces.items():
        ox1, oy1, ox2, oy2 = data["bbox"]
        ocx = (ox1 + ox2) // 2
        ocy = (oy1 + oy2) // 2

        dist = ((cx - ocx)**2 + (cy - ocy)**2)**0.5

        if dist < 80:
            tracked_faces[fid]["bbox"] = (x1, y1, x2, y2)
            return fid

    face_id_counter += 1

    tracked_faces[face_id_counter] = manager.dict({
        "bbox": (x1, y1, x2, y2),
        "nom": None,
        "distancia": None,
        "success": False,
        "img": None,
        "last_sent": 0,
        "history": []
    })

    return face_id_counter

# ---------------- HUD ----------------
def draw_hud_box(img, x, y, w, h, color):
    l = int(min(w, h) * 0.25)

    for t in [4, 1]:
        cv2.line(img, (x, y), (x + l, y), color, t)
        cv2.line(img, (x, y), (x, y + l), color, t)

        cv2.line(img, (x + w, y), (x + w - l, y), color, t)
        cv2.line(img, (x + w, y), (x + w, y + l), color, t)

        cv2.line(img, (x, y + h), (x + l, y + h), color, t)
        cv2.line(img, (x, y + h), (x, y + h - l), color, t)

        cv2.line(img, (x + w, y + h), (x + w - l, y + h), color, t)
        cv2.line(img, (x + w, y + h), (x + w, y + h - l), color, t)

# ---------------- VIDEO ----------------
def generate_frames():
    global tracked_faces

    last_time = 0

    while True:
        now = time.time()

        if now - last_time < FRAME_TIME:
            continue

        last_time = now

        success, frame = camera.read()
        if not success:
            continue

        frame = cv2.resize(frame,
            (FRAME_WIDTH, int(frame.shape[0] * FRAME_WIDTH / frame.shape[1]))
        )

        results = model(frame, verbose=False)
        current_ids = []

        if results and results[0].boxes is not None:
            for box in results[0].boxes.xyxy:
                x1, y1, x2, y2 = map(int, box)

                fid = get_face_id(x1, y1, x2, y2)
                current_ids.append(fid)

                data = tracked_faces.get(fid, {})

                color = (0, 255, 0) if data.get("success") else (0, 0, 255)

                draw_hud_box(frame, x1, y1, x2 - x1, y2 - y1, color)

                if data.get("nom"):
                    cv2.putText(frame, data["nom"],
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2
                    )

                if not face_queue.full():
                    last_sent = data.get("last_sent", 0)

                    if time.time() - last_sent > 2:
                        face_queue.put((frame.copy(), x1, y1, x2, y2, fid))
                        tracked_faces[fid]["last_sent"] = time.time()

        for fid in list(tracked_faces.keys()):
            if fid not in current_ids:
                del tracked_faces[fid]

        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            continue

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" +
               buffer.tobytes() +
               b"\r\n")

# ---------------- ROUTES ----------------
@app.route("/")
def index():
    return render_template("index2.html")

@app.route("/video")
def video():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/face_data")
def face_data():
    result = []

    for fid, data in tracked_faces.items():
        if data.get("nom"):
            result.append({
                "nom": data.get("nom"),
                "distancia": data.get("distancia"),
                "success": data.get("success"),
                "img": data.get("img"),
                "edat": data.get("edat"),
                "rol": data.get("rol")
            })

    return jsonify({"persones": result})

# ---------------- MAIN ----------------
if __name__ == "__main__":
    freeze_support()

    manager = Manager()
    tracked_faces = manager.dict()

    print("🔄 carregant embeddings...")
    db_embeddings = carregar_base_dades()

    p = Process(
        target=worker_reconeixement,
        args=(face_queue, tracked_faces, db_embeddings),
        daemon=True
    )
    p.start()

    try:
        app.run(host="0.0.0.0", port=5000)
    finally:
        face_queue.put(None)
        p.join()
        camera.release()