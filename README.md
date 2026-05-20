# 🧠 Face Recognition TDR

> Sistema de reconeixement facial en temps real amb intel·ligència artificial, visió per computador i deep learning.

---

## 📖 Descripció

Aquest projecte consisteix en el desenvolupament d’un sistema de reconeixement facial capaç de detectar i identificar persones en temps real a través d’una càmera.  
El sistema integra models avançats de detecció facial i embeddings facials per aconseguir identificacions eficients i precises.

---

# ✨ Funcionalitats

- ✅ Detecció de cares en temps real amb **YOLOv8**
- ✅ Reconeixement facial amb **DeepFace (ArcFace)**
- ✅ Comparació facial mitjançant embeddings
- ✅ Càlcul de similitud amb distància cosinus
- ✅ Tracking de cares entre fotogrames
- ✅ Sistema d’historial per estabilitzar resultats
- ✅ Classificació:
  - 🟢 Identificat
  - 🟡 Dubtós
  - 🔴 Desconegut
- ✅ Interfície web en temps real amb **Flask**

---

# 🛠️ Tecnologies utilitzades

| Tecnologia | Funció |
|------------|---------|
| Python | Llenguatge principal |
| OpenCV | Processament d’imatge |
| YOLOv8 | Detecció facial |
| DeepFace | Reconeixement facial |
| ArcFace | Extracció d’embeddings |
| Flask | Interfície web |
| NumPy | Operacions matemàtiques |
| Multiprocessing | Processament paral·lel |

---

# ⚙️ Funcionament del sistema

El sistema segueix aquest pipeline:

```text
📷 Captura de vídeo
        ↓
🔍 Detecció facial amb YOLOv8
        ↓
🧠 Extracció d’embeddings (ArcFace)
        ↓
📊 Comparació amb la base de dades
        ↓
📐 Distància cosinus
        ↓
✅ Identificació final
        ↓
🔁 Filtratge amb historial
        ↓
🌐 Visualització web
```

---

# 📂 Estructura del projecte

```bash
face-recognition-tdr/
│
├── main.py
├── requirements.txt
├── database/
├── models/
├── static/
├── templates/
└── README.md
```

---

# 🚀 Instal·lació

## 1️⃣ Clonar el repositori

```bash
git clone https://github.com/J4-pro/face-recognition-tdr.git
cd face-recognition-tdr
```

## 2️⃣ Instal·lar dependències

```bash
pip install -r requirements.txt
```

## 3️⃣ Executar el projecte

```bash
python main.py
```

---

# 🌐 Interfície Web

La interfície desenvolupada amb Flask permet:

- 🎥 Veure el vídeo en directe
- 🧠 Visualitzar identificacions
- 📊 Mostrar l’estat de cada detecció
- ⚡ Monitoritzar el sistema en temps real

---

# 🧪 Resultats

| Aspecte | Resultat |
|----------|-----------|
| Precisió aproximada | ~80% |
| Temps real | ✅ |
| Compatible amb PC normals | ✅ |
| Tracking facial | ✅ |

---

# 📸 Demo

Afegeix aquí captures o GIFs del projecte:

```md
![Demo](images/demo.png)
```

---

# 📈 Futures millores

- 🔥 Optimització amb GPU
- 👥 Suport multiusuari
- 📷 Compatibilitat amb múltiples càmeres
- 📊 Dashboard amb estadístiques
- ☁️ Integració amb base de dades remota

---

# 👨‍💻 Autor

**Treball de Recerca (TDR)**  
Projecte desenvolupat amb Python i tecnologies de visió artificial.

---

# 📄 Llicència

Aquest projecte està sota la llicència **MIT**.
````
