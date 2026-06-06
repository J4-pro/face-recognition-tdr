import os
import time
import cv2
import pickle
import requests
import numpy as np

from flask import Flask, Response, render_template, jsonify, redirect
from ultralytics import YOLO
from multiprocessing import Process, Queue, Manager, freeze_support
from deepface import DeepFace
from collections import Counter

from persones import persones

# =========================================================
# CONFIG
# =========================================================

FPS = 30
FRAME_TIME = 1.0 / FPS

FRAME_WIDTH = 640

THRESHOLD = 0.5
SS
EMBEDDINGS_FILE = "models/embeddings.pkl"

RASPBERRY_IP = "http://192.168.4.17:5001"

DETECTION_SKIP = 2

# =========================================================
# FLASK
# =========================================================

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True

# =========================================================
# MODELS
# =========================================================

model = YOLO("models/yolov8n-face.pt")

camera = cv2.VideoCapture(0)

# =========================================================
# MULTIPROCESS
# =========================================================

face_queue = Queue(maxsize=5)

manager = None
tracked_faces = None

face_id_counter = 0

# =========================================================
# EMBEDDINGS
# =========================================================

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

    return 1 - np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )


# =========================================================
# LOAD DATABASE
# =========================================================

def carregar_base_dades():

    if not os.path.exists(EMBEDDINGS_FILE):

        print("❌ embeddings.pkl no trobat")
        exit()

    with open(EMBEDDINGS_FILE, "rb") as f:

        db = pickle.load(f)

    print("✅ Embeddings carregats:", db.keys())

    return db


# =========================================================
# WORKER RECONEIXEMENT
# =========================================================

def worker_reconeixement(queue, tracked_faces, db_embeddings):

    while True:

        data = queue.get()

        if data is None:
            break

        face_img, fid = data

        if face_img.size == 0:
            continue

        emb = get_embedding(face_img)

        if emb is None:
            continue

        distancies = []

        # comparar embeddings
        for persona, emb_list in db_embeddings.items():

            for emb_db in emb_list:

                dist = cosine_distance(emb, emb_db)

                distancies.append((persona, dist))

        if not distancies:
            continue

        distancies.sort(key=lambda x: x[1])

        millor_nom, millor_dist = distancies[0]

        # =================================================
        # DECISIÓ
        # =================================================

        if millor_dist > 0.6:

            millor_nom = "unknown"

        elif millor_dist > THRESHOLD:

            millor_nom = "doubtful"

        print(f"[{fid}] {millor_nom} ({millor_dist:.3f})")

        # =================================================
        # HISTORIAL
        # =================================================

        if fid in tracked_faces:

            data_face = tracked_faces[fid]

            hist = data_face.get("history", [])

            hist.append(millor_nom)

            if len(hist) > 5:
                hist.pop(0)

            noms_valids = [
                n for n in hist
                if n not in ["unknown", "doubtful"]
            ]

            if noms_valids:

                final = Counter(noms_valids).most_common(1)[0][0]

            else:

                if "doubtful" in hist:
                    final = "doubtful"
                else:
                    final = "unknown"

            info = persones.get(final, {})

            # actualitzar
            data_face.update({

                "nom": final,
                "distancia": float(millor_dist),
                "success": final not in ["unknown", "doubtful"],

                "history": hist,

                "edat": info.get("edat"),
                "rol": info.get("rol"),
                "img": info.get("imatge_ref")

            })

            tracked_faces[fid] = data_face


# =========================================================
# TRACKING SIMPLE
# =========================================================

def get_face_id(x1, y1, x2, y2):

    global face_id_counter
    global tracked_faces

    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    for fid, data in tracked_faces.items():

        ox1, oy1, ox2, oy2 = data["bbox"]

        ocx = (ox1 + ox2) // 2
        ocy = (oy1 + oy2) // 2

        dist = ((cx - ocx) ** 2 + (cy - ocy) ** 2) ** 0.5

        if dist < 80:

            data["bbox"] = (x1, y1, x2, y2)
            data["last_seen"] = time.time()

            tracked_faces[fid] = data

            return fid

    # =====================================================
    # NOVA CARA
    # =====================================================

    face_id_counter += 1

    tracked_faces[face_id_counter] = {

        "bbox": (x1, y1, x2, y2),

        "nom": None,
        "distancia": None,

        "success": False,

        "img": None,

        "last_sent": 0,

        "history": [],

        "last_seen": time.time(),

        "edat": None,
        "rol": None
    }

    return face_id_counter


# =========================================================
# HUD
# =========================================================

def draw_hud_box(img, x, y, w, h, color):

    l = int(min(w, h) * 0.25)

    thickness = 3

    # top left
    cv2.line(img, (x, y), (x + l, y), color, thickness)
    cv2.line(img, (x, y), (x, y + l), color, thickness)

    # top right
    cv2.line(img, (x + w, y), (x + w - l, y), color, thickness)
    cv2.line(img, (x + w, y), (x + w, y + l), color, thickness)

    # bottom left
    cv2.line(img, (x, y + h), (x + l, y + h), color, thickness)
    cv2.line(img, (x, y + h), (x, y + h - l), color, thickness)

    # bottom right
    cv2.line(img, (x + w, y + h), (x + w - l, y + h), color, thickness)
    cv2.line(img, (x + w, y + h), (x + w, y + h - l), color, thickness)


# =========================================================
# VIDEO STREAM
# =========================================================

def generate_frames():

    global tracked_faces

    frame_count = 0

    last_time = 0

    results = None

    while True:

        now = time.time()

        # limitar FPS
        if now - last_time < FRAME_TIME:

            time.sleep(0.001)
            continue

        last_time = now

        success, frame = camera.read()

        if not success:
            continue

        # resize
        frame = cv2.resize(
            frame,
            (
                FRAME_WIDTH,
                int(frame.shape[0] * FRAME_WIDTH / frame.shape[1])
            )
        )

        frame_count += 1

        # =================================================
        # YOLO
        # =================================================

        if frame_count % DETECTION_SKIP == 0:

            results = model(frame, verbose=False)

        current_ids = []

        if results and results[0].boxes is not None:

            for box in results[0].boxes.xyxy:

                x1, y1, x2, y2 = map(int, box)

                fid = get_face_id(x1, y1, x2, y2)

                current_ids.append(fid)

                data = tracked_faces.get(fid, {})

                # =========================================
                # COLOR
                # =========================================

                if data.get("nom") == "doubtful":

                    color = (0, 165, 255)

                elif data.get("success"):

                    color = (0, 255, 0)

                else:

                    color = (0, 0, 255)

                # =========================================
                # DIBUIX
                # =========================================

                draw_hud_box(
                    frame,
                    x1,
                    y1,
                    x2 - x1,
                    y2 - y1,
                    color
                )

                if data.get("nom"):

                    cv2.putText(
                        frame,
                        data["nom"],
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2
                    )

                # =========================================
                # ENVIAR AL WORKER
                # =========================================

                if not face_queue.full():

                    last_sent = data.get("last_sent", 0)

                    if time.time() - last_sent > 2:

                        face_crop = frame[y1:y2, x1:x2].copy()

                        face_queue.put((face_crop, fid))

                        data["last_sent"] = time.time()

                        tracked_faces[fid] = data

        # =================================================
        # ELIMINAR CARES ANTIGUES
        # =================================================

        for fid in list(tracked_faces.copy().keys()):

            if fid not in current_ids:

                data = tracked_faces[fid]

                if time.time() - data.get("last_seen", time.time()) > 1:

                    del tracked_faces[fid]

        # =================================================
        # FPS
        # =================================================

        fps = int(1 / max(time.time() - now, 0.001))

        cv2.putText(
            frame,
            f"FPS: {fps}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2
        )

        # =================================================
        # STREAM
        # =================================================

        ret, buffer = cv2.imencode(".jpg", frame)

        if not ret:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            buffer.tobytes() +
            b"\r\n"
        )


# =========================================================
# ROUTES
# =========================================================

@app.route("/")
def index():

    return render_template("index2.html")


@app.route("/video")
def video():

    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


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

    result.sort(key=lambda x: x["nom"])

    return jsonify({"persones": result})


# =========================================================
# SERVO
# =========================================================

@app.route("/dreta")
def dreta():

    try:
        requests.get(
            RASPBERRY_IP + "/dreta",
            timeout=1
        )
    except:
        pass

    return ('', 204)


@app.route("/esquerra")
def esquerra():

    try:
        requests.get(
            RASPBERRY_IP + "/esquerra",
            timeout=1
        )
    except:
        pass

    return ('', 204)

# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    freeze_support()

    manager = Manager()

    tracked_faces = manager.dict()

    print("🔄 carregant embeddings...")

    db_embeddings = carregar_base_dades()

    p = Process(

        target=worker_reconeixement,

        args=(
            face_queue,
            tracked_faces,
            db_embeddings
        ),

        daemon=True
    )

    p.start()

    try:

        app.run(
            host="0.0.0.0",
            port=5000,
            threaded=True
        )

    finally:

        face_queue.put(None)

        p.join()

        camera.release()