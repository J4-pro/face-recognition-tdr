# Face Recognition TDR

Sistema de reconeixement facial en temps real amb seguiment automàtic mitjançant una càmera controlada per servo i una Raspberry Pi.

## Descripció

Aquest projecte combina visió artificial, intel·ligència artificial i sistemes encastats per crear una plataforma capaç de:

* Detectar cares en temps real utilitzant YOLOv8.
* Reconèixer persones mitjançant embeddings facials generats amb DeepFace (ArcFace).
* Mostrar informació associada a cada persona identificada.
* Controlar automàticament la posició d'una càmera muntada sobre un servo motor.
* Comunicar un ordinador i una Raspberry Pi a través de Wi-Fi.

L'ordinador realitza tota la càrrega de processament (detecció i reconeixement facial), mentre que la Raspberry Pi s'encarrega únicament del control físic del servo.

---

## Arquitectura del sistema

```text
                           WIFI
┌───────────────────────────────────────────┐
│                                           │
│   Ordinador Principal                     │
│                                           │
│  ┌────────────────────────────────────┐   │
│  │ Flask Web Server                   │   │
│  │ YOLOv8 Face Detection              │   │
│  │ DeepFace Recognition               │   │
│  │ Gestió d'Embeddings                │   │
│  └────────────────────────────────────┘   │
│                                           │
└───────────────┬───────────────────────────┘
                │
                │ HTTP Requests
                ▼
┌───────────────────────────────────────────┐
│ Raspberry Pi                              │
│                                           │
│ Flask Server                              │
│ GPIOZero                                  │
│ Control del Servo                         │
└───────────────┬───────────────────────────┘
                │
                ▼
          Servo Motor
                │
                ▼
            Càmera
```

---

## Tecnologies utilitzades

### Visió artificial

* OpenCV
* YOLOv8 Face Detection
* DeepFace
* ArcFace

### Backend

* Flask
* Multiprocessing

### Hardware

* Raspberry Pi
* Servo Motor
* Càmera USB

### Comunicació

* HTTP REST
* Xarxa Wi-Fi

---

## Estructura del projecte

```text
FACIAL_RECOGNITION/
│
├── models/
│   ├── embeddings.pkl
│   ├── yolov8n-face.pt
│   └── yolov8s-face.pt
│
├── faces/
├── images/
├── static/
├── templates/
│   └── index2.html
│
├── capture_faces.py
├── create_embeddings.py
├── main.py
├── raspi.py
├── persones.py
│
├── requirements.txt
└── README.md
```

---

## Funcionament

### 1. Captura de vídeo

La càmera USB envia vídeo en temps real a l'ordinador.

### 2. Detecció facial

El model YOLOv8 detecta les cares presents en cada fotograma.

### 3. Tracking

Cada cara detectada rep un identificador temporal per fer-ne el seguiment entre fotogrames.

### 4. Reconeixement facial

Les cares detectades s'envien a un procés independent que:

* Genera embeddings amb ArcFace.
* Compara els embeddings amb la base de dades.
* Determina la identitat de la persona.

### 5. Visualització

La interfície web mostra:

* Nom.
* Rol.
* Edat.
* Imatge de referència.
* Estat del reconeixement.

### 6. Control del servo

Quan la cara es desplaça dins la imatge:

* El PC envia una petició HTTP a la Raspberry Pi.
* La Raspberry Pi modifica la posició del servo.
* El servo orienta la càmera cap a la persona.

---

## Creació de la base de dades facial

### Capturar imatges

```bash
python capture_faces.py
```

Les imatges es desen dins la carpeta:

```text
faces/
```

### Generar embeddings

```bash
python create_embeddings.py
```

Es crearà:

```text
models/embeddings.pkl
```

Aquest fitxer conté els vectors facials de totes les persones registrades.

---

## Instal·lació

### Ordinador principal

Instal·lar dependències:

```bash
pip install -r requirements.txt
```

Descarregar els models:

```text
models/yolov8n-face.pt
models/yolov8s-face.pt
```

---

### Raspberry Pi

Instal·lar:

```bash
pip install flask gpiozero
```

Executar:

```bash
python raspi.py
```

El servidor quedarà disponible al port:

```text
5001
```

---

## Execució

### Raspberry Pi

```bash
python raspi.py
```

### Ordinador

```bash
python main.py
```

Accedir des del navegador:

```text
http://localhost:5000
```

---

## Comunicació PC ↔ Raspberry

El sistema utilitza peticions HTTP.

### Moure a la dreta

```http
GET /dreta
```

### Moure a l'esquerra

```http
GET /esquerra
```

Aquestes peticions són enviades automàticament des de l'ordinador quan cal reajustar la posició de la càmera.

---

## Característiques principals

* Reconeixement facial en temps real.
* Detecció amb YOLOv8.
* Embeddings facials amb ArcFace.
* Interfície web amb Flask.
* Processament multiprocés.
* Seguiment de persones.
* Control remot de servo per Wi-Fi.
* Sistema distribuït entre PC i Raspberry Pi.

---

## Autor

**J4-pro**

Treball de Recerca (TDR) sobre reconeixement facial i seguiment automàtic de persones mitjançant visió artificial i sistemes embeguts.
