# Face Recognition TDR

Sistema de reconeixement facial en temps real amb seguiment automàtic mitjançant una càmera controlada per una Raspberry Pi i un servo motor.

## Introducció

Aquest projecte ha estat desenvolupat com a Treball de Recerca (TDR) i combina visió artificial, intel·ligència artificial i sistemes encastats per crear un sistema capaç de detectar, identificar i seguir persones automàticament.

L'arquitectura es divideix en dos dispositius:

* **Ordinador principal**: realitza la detecció i el reconeixement facial.
* **Raspberry Pi**: controla el servo motor que orienta la càmera.

Tots dos dispositius es comuniquen a través d'una xarxa Wi-Fi mitjançant peticions HTTP.

---

# Funcionament del sistema

1. La càmera USB envia vídeo en temps real a l'ordinador.
2. El model YOLOv8 detecta les cares presents a la imatge.
3. DeepFace genera embeddings facials de cada cara.
4. Els embeddings es comparen amb una base de dades pròpia.
5. Si una persona és reconeguda, es mostra la seva informació.
6. Quan la persona es desplaça lateralment, l'ordinador envia una ordre a la Raspberry Pi.
7. La Raspberry Pi mou el servo perquè la càmera continuï apuntant cap a la persona.

---

# Arquitectura

```text
                         WIFI

┌────────────────────────────────────┐
│        Ordinador Principal         │
│                                    │
│  Flask                            │
│  OpenCV                           │
│  YOLOv8 Face Detection            │
│  DeepFace (ArcFace)               │
│  Base de dades d'embeddings       │
└───────────────┬────────────────────┘
                │
                │ HTTP
                ▼
┌────────────────────────────────────┐
│          Raspberry Pi              │
│                                    │
│ Flask                             │
│ GPIOZero                          │
│ Control del Servo                 │
└───────────────┬────────────────────┘
                │
                ▼
           Servo Motor
                │
                ▼
             Càmera
```

---

# Tecnologies utilitzades

## Intel·ligència Artificial

* DeepFace
* ArcFace

## Visió Artificial

* OpenCV
* YOLOv8

## Backend

* Flask
* Multiprocessing

## Hardware

* Raspberry Pi
* Servo Motor
* Càmera USB

## Comunicació

* HTTP REST
* Xarxa Wi-Fi

---

# Estructura del projecte

```text
FACIAL_RECOGNITION/
│
├── faces/
├── images/
├── models/
│   ├── embeddings.pkl
│   ├── yolov8n-face.pt
│   └── yolov8s-face.pt
│
├── static/
├── templates/
│   └── index2.html
│
├── capture_faces.py
├── create_embeddings.py
├── main.py
├── persones.py
├── raspi.py
│
├── requirements-ordinador_principal.txt
├── requirements-raspberry.txt
│
└── README.md
```

---

# Fitxers principals

## main.py

Programa principal del sistema.

Funcions:

* Captura de vídeo.
* Detecció de cares.
* Tracking de persones.
* Reconeixement facial.
* Interfície web.
* Comunicació amb la Raspberry Pi.

---

## raspi.py

Servidor Flask executat a la Raspberry Pi.

Funcions:

* Recepció d'ordres HTTP.
* Control del servo.
* Moviment esquerra/dreta.

---

## capture_faces.py

Permet capturar fotografies de les persones que es volen registrar al sistema.

Les imatges es desen a:

```text
faces/
```

---

## create_embeddings.py

Genera els embeddings facials de totes les persones registrades.

Crea el fitxer:

```text
models/embeddings.pkl
```

---

## persones.py

Base de dades amb informació addicional de cada persona:

* Nom
* Edat
* Rol
* Imatge de referència

---

# Instal·lació

## 1. Clonar el repositori

```bash
git clone https://github.com/J4-pro/face-recognition-tdr.git
cd face-recognition-tdr
```

---

## 2. Configurar l'ordinador principal

Instal·lar dependències:

```bash
pip install -r requirements-ordinador_principal.txt
```

---

## 3. Configurar la Raspberry Pi

Instal·lar dependències:

```bash
pip install -r requirements-raspberry.txt
```

---

# Configuració de la Raspberry Pi

Al fitxer `main.py` cal configurar la IP de la Raspberry:

```python
RASPBERRY_IP = "http://192.168.X.X:5001"
```

Substituir-la per la IP real de la Raspberry Pi.

---

# Execució

## Raspberry Pi

Iniciar el servidor:

```bash
python raspi.py
```

El servidor quedarà escoltant al port:

```text
5001
```

---

## Ordinador principal

Executar:

```bash
python main.py
```

La interfície web quedarà disponible a:

```text
http://localhost:5000
```

---

# Procés de registre de persones

### Capturar fotografies

```bash
python capture_faces.py
```

### Generar embeddings

```bash
python create_embeddings.py
```

### Executar el sistema

```bash
python main.py
```

---

# Característiques

* Detecció facial en temps real.
* Reconeixement facial amb DeepFace.
* Tracking de persones.
* Seguiment automàtic mitjançant servo.
* Comunicació PC ↔ Raspberry per Wi-Fi.
* Interfície web amb Flask.
* Arquitectura distribuïda.
* Sistema modular i ampliable.

---

# Autor

**J4-pro**

Treball de Recerca (TDR)

Reconeixement facial i seguiment automàtic de persones mitjançant visió artificial, intel·ligència artificial i sistemes encastats.
