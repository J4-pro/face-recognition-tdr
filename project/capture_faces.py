import cv2
import os
from ultralytics import YOLO

BASE_DIR = os.path.dirname(__file__)
YOLO_PATH = os.path.join(BASE_DIR, "model", "yolov8n-face.pt")

model = YOLO(YOLO_PATH)

cam_index = int(input("📹 Camara (0, 1, ...): "))
cap = cv2.VideoCapture(cam_index)

print("🚀 Captura amb YOLO iniciada")
print("Prem ESPAI per guardar una foto")
print("Escriu 'exit' per sortir\n")

while True:
    PERSON_NAME = input("👤 Nom de la persona (exit per sortir): ").strip()

    if PERSON_NAME.lower() == "exit":
        break

    output_dir = os.path.join(BASE_DIR, "dataset", PERSON_NAME)
    os.makedirs(output_dir, exist_ok=True)

    # Comptador de fotos existents (per continuar numeració)
    existing = len(os.listdir(output_dir))
    count = existing

    print(f"📁 Carpeta: {output_dir}")
    print("📸 Fes tantes fotos com vulguis. Prem ESPAI per capturar.")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        results = model(frame)[0]

        face_detected = False

        if results.boxes is not None and len(results.boxes) > 0:
            box = results.boxes[0]

            if box.conf[0] >= 0.5:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                margin = 30
                x1 = max(0, x1 - margin)
                y1 = max(0, y1 - margin)
                x2 = x2 + margin
                y2 = y2 + margin

                face = frame[y1:y2, x1:x2]

                if face is not None and face.size > 0:
                    if face.shape[0] >= 80 and face.shape[1] >= 80:
                        face_detected = True
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

        if not face_detected:
            cv2.putText(frame, "NO FACE", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        cv2.putText(frame, f"{PERSON_NAME}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

        cv2.imshow("FaceID YOLO", frame)

        key = cv2.waitKey(1)

        if key == 27:  # ESC
            cap.release()
            cv2.destroyAllWindows()
            exit()

        if key == 32 and face_detected:  # ESPAI
            face_clean = cv2.resize(face, (224, 224))
            filename = os.path.join(output_dir, f"{count}.png")
            cv2.imwrite(filename, face_clean)
            print("✅ guardat:", filename)
            count += 1

        if key == ord('n'):  # passar a una altra persona
            print("\n➡️ Canviant de persona...\n")
            break

cap.release()
cv2.destroyAllWindows()
print("✅ FINAL")
