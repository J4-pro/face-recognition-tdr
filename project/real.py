import cv2
import os
import pickle
import numpy as np
import time
import requests
import base64

from flask import Flask, render_template, jsonify
from flask_sock import Sock
from deepface import DeepFace
from ultralytics import YOLO
from threading import Thread, Lock
from queue import Queue

# ==============================
# CONFIGURACIÓ
# ==============================

BASE_DIR = os.path.dirname(__file__)

YOLO_PATH = os.path.join(BASE_DIR, "model", "yolov8n-face.pt")
EMBED_PATH = os.path.join(BASE_DIR, "model", "embeddings.pkl")

MODEL_NAME = "ArcFace"
FRAME_W = 416

RASPBERRY_PI = input("🌐 IP del Raspberry Pi (sense http://): ").strip()
cam_index_input = input("📷 Índex de la càmera (0, 1, ...): ").strip()

cam_index = int(cam_index_input) if cam_index_input else 0

DEBUG = False

# ==============================
# PERSONES
# ==============================

from persones import persones

# ==============================
# CARREGAR EMBEDDINGS
# ==============================

with open(EMBED_PATH, "rb") as f:
    EMBEDDINGS_DB = pickle.load(f)
    print(f"✅ Carregat {len(EMBEDDINGS_DB)} embeddings")

for k in EMBEDDINGS_DB:
    EMBEDDINGS_DB[k] = EMBEDDINGS_DB[k] / np.linalg.norm(EMBEDDINGS_DB[k])

# ==============================
# FLASK + WEBSOCKET
# ==============================

app = Flask(__name__)
sock = Sock(app)

detections = []
detections_lock = Lock()

faces_state = {}   # face_id → {name, center, last_try, last_confirm, ever_identified, last_seen}
faces_lock = Lock()

RETRY_TIME = 1.0
REVALIDATE_TIME = 5.0
THRESHOLD = 0.80

COLOR_UNKNOWN = (0, 0, 255)
COLOR_KNOWN = (0, 255, 0)

face_queue = Queue(maxsize=20)

TARGET_FPS = 15
YOLO_SKIP_FRAMES = 3

# ==============================
# FUNCIONS
# ==============================

def crop_face(frame, box, margin=30):
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    h, w = frame.shape[:2]

    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(w, x2 + margin)
    y2 = min(h, y2 + margin)

    if x2 <= x1 or y2 <= y1:
        return None

    face = frame[y1:y2, x1:x2]

    if face.shape[0] < 120 or face.shape[1] < 120:
        return None

    return face


def get_embedding(face_img):
    try:
        face = cv2.resize(face_img, (224, 224))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        emb = DeepFace.represent(
            img_path=face,
            model_name=MODEL_NAME,
            enforce_detection=False,
            detector_backend="skip"
        )

        emb = np.array(emb[0]["embedding"])
        return emb / np.linalg.norm(emb)

    except:
        return None


def recognize_face(embedding):
    best_name = "unknown"
    best_dist = 999

    for name, ref_emb in EMBEDDINGS_DB.items():
        dist = np.linalg.norm(embedding - ref_emb)
        if dist < best_dist:
            best_dist = dist
            best_name = name

    return (best_name, best_dist) if best_dist < THRESHOLD else ("unknown", best_dist)


def deepface_worker():

    while True:

        face_id, face_img = face_queue.get()

        try:
            emb = get_embedding(face_img)
            if emb is None:
                continue

            new_name, dist = recognize_face(emb)
            now = time.time()

            with faces_lock:

                if face_id not in faces_state:
                    continue

                if new_name != "unknown":
                    faces_state[face_id]["name"] = new_name
                    faces_state[face_id]["last_confirm"] = now
                    faces_state[face_id]["ever_identified"] = True

                else:
                    if not faces_state[face_id]["ever_identified"]:
                        faces_state[face_id]["name"] = "unknown"

        except Exception as e:
            print(f"[DeepFace Worker Error] {e}")

        finally:
            face_queue.task_done()


def draw_hud_box(img, x, y, w, h, color):

    thickness = 2
    corner_len = int(min(w, h) * 0.25)

    # EXTREMS (com tenies, però més nets)

    # top left
    cv2.line(img, (x, y), (x + corner_len, y), color, thickness)
    cv2.line(img, (x, y), (x, y + corner_len), color, thickness)

    # top right
    cv2.line(img, (x + w, y), (x + w - corner_len, y), color, thickness)
    cv2.line(img, (x + w, y), (x + w, y + corner_len), color, thickness)

    # bottom left
    cv2.line(img, (x, y + h), (x + corner_len, y + h), color, thickness)
    cv2.line(img, (x, y + h), (x, y + h - corner_len), color, thickness)

    # bottom right
    cv2.line(img, (x + w, y + h), (x + w - corner_len, y + h), color, thickness)
    cv2.line(img, (x + w, y + h), (x + w, y + h - corner_len), color, thickness)

    # 🔥 LÍNIA CENTRAL HORITZONTAL (HUD style)
    cv2.line(img, (x, y + h//2), (x + w, y + h//2), color, 1)

    # 🔥 PUNT CENTRAL
    cx = x + w // 2
    cy = y + h // 2
    cv2.circle(img, (cx, cy), 3, color, -1)

    # 🔥 MARQUES VERTICALS PETITES
    cv2.line(img, (cx, y), (cx, y + 10), color, 1)
    cv2.line(img, (cx, y + h), (cx, y + h - 10), color, 1)


# ==============================
# YOLO
# ==============================

model = YOLO(YOLO_PATH)
print("✅ YOLO carregat")

print("🔥 Escalfant ArcFace...")

dummy = np.zeros((224,224,3), dtype=np.uint8)

try:
    DeepFace.represent(
        img_path=dummy,
        model_name=MODEL_NAME,
        enforce_detection=False,
        detector_backend="skip"
    )
except:
    pass

print("✅ ArcFace preparat")

# ==============================
# RUTA PRINCIPAL
# ==============================

@app.route("/")
def index():
    return render_template("index.html")

# ==============================
# WEBSOCKET DE VÍDEO
# ==============================

@sock.route("/ws")
def ws_video(ws):

    global detections

    cap = cv2.VideoCapture(cam_index)

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("❌ No s'ha pogut obrir la càmera")
        return

    print("🎥 Streaming iniciat...")

    frame_counter = 0
    last_send = 0
    cached_results = None

    try:

        while True:

            ret, frame = cap.read()
            if not ret:
                continue

            frame_counter += 1
            now = time.time()

            # YOLO SKIP
            if cached_results is None or frame_counter % YOLO_SKIP_FRAMES == 0:
                cached_results = model.track(
                    frame,
                    persist=True,
                    imgsz=416,
                    verbose=False
                )[0]

            results = cached_results
            if results is None:
                continue

            boxes = results.boxes
            new_detections = []
            seen_labels = set()

            # NETEJA ESTATS ANTICS
            with faces_lock:
                to_delete = [
                    fid for fid, data in faces_state.items()
                    if now - data.get("last_seen", now) > 10
                ]
                for fid in to_delete:
                    del faces_state[fid]

            # PROCESSAR CARES
            for box in boxes:

                if box.id is None:
                    continue

                track_id = int(box.id.item())
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                with faces_lock:

                    if track_id not in faces_state:
                        faces_state[track_id] = {
                            "name": "unknown",
                            "center": (cx, cy),
                            "last_try": 0,
                            "last_confirm": 0,
                            "ever_identified": False,
                            "last_seen": now,
                        }
                    else:
                        faces_state[track_id]["center"] = (cx, cy)
                        faces_state[track_id]["last_seen"] = now

                    state = faces_state[track_id].copy()

                face_img = crop_face(frame, box)
                if face_img is None:
                    continue

                name = state["name"]
                ever_identified = state["ever_identified"]

                # DEEPFACE LOGIC CORREGIDA
                need_check = False

                if name == "unknown" and not ever_identified:
                    need_check = now - state["last_try"] > RETRY_TIME

                elif name != "unknown":
                    need_check = now - state["last_confirm"] > REVALIDATE_TIME

                elif name == "unknown" and ever_identified:
                    need_check = now - state["last_try"] > REVALIDATE_TIME

                if need_check and not face_queue.full():
                    face_queue.put((track_id, face_img.copy()))
                    with faces_lock:
                        if track_id in faces_state:
                            faces_state[track_id]["last_try"] = now

                # DIBUIXAR
                color = COLOR_KNOWN if name != "unknown" else COLOR_UNKNOWN

                draw_hud_box(frame, x1, y1, x2 - x1, y2 - y1, color)
                cv2.putText(
                    frame, name, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
                )

                # SIDEBAR
                if name not in seen_labels:
                    seen_labels.add(name)

                    if name in persones:
                        info = persones[name]
                        new_detections.append({
                            "nom": name,
                            "edat": info["edat"],
                            "rol": info["rol"],
                            "imatge": "/static/" + info["imatge_ref"]
                        })
                    else:
                        new_detections.append({
                            "nom": "unknown",
                            "edat": "-",
                            "rol": "-",
                            "imatge": "/static/persones/Unknown.png"
                        })

            # ACTUALITZAR DETECTIONS AMB LOCK
            with detections_lock:
                detections = new_detections

            # LIMITADOR FPS
            if now - last_send < (1 / TARGET_FPS):
                continue

            last_send = now

            frame_small = cv2.resize(frame, (FRAME_W, FRAME_W))

            success, buffer = cv2.imencode(
                ".jpg",
                frame_small,
                [int(cv2.IMWRITE_JPEG_QUALITY), 70]
            )

            if not success:
                continue

            try:
                ws.send(base64.b64encode(buffer).decode("utf-8"))
            except:
                break

    finally:
        cap.release()
        print("📷 Camera alliberada")

# ==============================
# SIDEBAR
# ==============================

@app.route("/face_data")
def face_data():
    with detections_lock:
        return jsonify(detections)

# ==============================
# SERVO
# ==============================

@app.route("/dreta")
def dreta():
    try:
        requests.get(f"http://{RASPBERRY_PI}/dreta", timeout=1)
    except:
        pass
    return ('', 204)

@app.route("/esquerra")
def esquerra():
    try:
        requests.get(f"http://{RASPBERRY_PI}/esquerra", timeout=1)
    except:
        pass
    return ('', 204)


# ==============================
# MAIN
# ==============================

if __name__ == "__main__":
    print("🚀 Servidor iniciat a http://0.0.0.0:5000")
    for _ in range(2):
        Thread(target=deepface_worker, daemon=True).start()
    app.run(host="0.0.0.0", port=5000)
