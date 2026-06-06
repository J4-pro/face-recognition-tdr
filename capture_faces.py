import cv2
import os
import time
from ultralytics import YOLO

PERSON_NAME = "Jan_Onieva"
NUM_PHOTOS = 20

output_dir = rf"C:\Users\JAN\PycharmProjects\Facial_recognition\faces\{PERSON_NAME}"
os.makedirs(output_dir, exist_ok=True)

print("📁 Guardant a:", output_dir)

model = YOLO("../models/yolov8s-face.pt")

cap = cv2.VideoCapture(1)  # prova 0 primer

if not cap.isOpened():
    print("❌ No es pot obrir la càmera")
    exit()

count = 0
last = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Error llegint càmera")
        break

    results = model(frame, verbose=False)

    for r in results:
        for box in r.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)

            h, w, _ = frame.shape
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue

            if time.time() - last > 1 and count < NUM_PHOTOS:
                # ✅ MANTENIR COLOR
                face = cv2.resize(face, (300, 300))

                filename = os.path.join(output_dir, f"{count + 1}.png")
                success = cv2.imwrite(filename, face)

                if success:
                    print("💾 Guardat a:", filename)
                    count += 1

                last = time.time()

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("Captura", frame)

    if cv2.waitKey(1) == 27 or count >= NUM_PHOTOS:
        break

cap.release()
cv2.destroyAllWindows()

print("🏁 Finalitzat. Fotos guardades:", count)