from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
import io, os
import torch

CONFIDENCE = 0.50
IMAGE_SIZE  = 640

# Force CPU on Render; keep threads minimal
DEVICE = "cpu"
torch.set_num_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"

# Serve static files directly from the repo root (where index.html is)
app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)

model = None
CLASS_NAMES = None

def get_model():
    """Lazy-load the YOLO model once per process."""
    global model, CLASS_NAMES
    if model is None:
        from ultralytics import YOLO
        # default to your repo file; allow override via env
        model_path = os.getenv("MODEL_PATH", "models/yolov8n.pt")
        model = YOLO(model_path)
        CLASS_NAMES = model.names
    return model

# ----------------- STATIC / UI -----------------
@app.get("/")
def home():
    # Serve your index.html at root
    return app.send_static_file("index.html")

# If you add other assets later (css/js), this serves them automatically
@app.get("/<path:filename>")
def static_files(filename):
    return send_from_directory(".", filename)

# ----------------- API -----------------
@app.get("/api/ping")
def ping():
    loaded = model is not None
    return jsonify({"status": "ok", "model_loaded": loaded, "device": DEVICE})

@app.post("/api/detect")
def detect():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No file field 'image'"}), 400
        f = request.files["image"]
        raw = f.read()
        if not raw:
            return jsonify({"error": "Empty file"}), 400

        img = Image.open(io.BytesIO(raw)).convert("RGB")

        m = get_model()
        results = m.predict(
            img,
            device=DEVICE,
            conf=CONFIDENCE,
            imgsz=IMAGE_SIZE,
            verbose=False
        )
        r = results[0]

        objects, counts = [], {}
        if getattr(r, "boxes", None) is not None and len(r.boxes) > 0:
            cls  = r.boxes.cls.detach().cpu().numpy().tolist()
            conf = r.boxes.conf.detach().cpu().numpy().tolist()
            for c, cf in sorted(zip(cls, conf), key=lambda x: x[1], reverse=True):
                label = CLASS_NAMES.get(int(c), f"class_{int(c)}")
                objects.append({"label": label, "confidence": float(cf)})
                counts[label] = counts.get(label, 0) + 1

        return jsonify({"objects": objects, "counts": counts})
    except Exception as e:
        # bubble up a readable error; check Render logs if this fires
        return jsonify({"error": str(e)}), 500
