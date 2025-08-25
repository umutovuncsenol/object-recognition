from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io, os
import torch

CONFIDENCE = 0.5
IMAGE_SIZE = 640

# keep CPU on Render free; also reduce threads for RAM/CPU
DEVICE = "cpu"
torch.set_num_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"

app = Flask(__name__)
CORS(app)

model = None
CLASS_NAMES = None

def get_model():
    global model, CLASS_NAMES
    if model is None:
        from ultralytics import YOLO
        # use tiny default if no file; avoids downloading big models
        model_path = os.getenv("MODEL_PATH", "yolov8n.pt")
        model = YOLO(model_path)
        CLASS_NAMES = model.names
    return model

@app.route("/api/ping")
def ping():
    return jsonify({"status": "ok"})

@app.route("/api/detect", methods=["POST"])
def detect():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No file field 'image'"}), 400
        raw = request.files["image"].read()
        if not raw:
            return jsonify({"error": "Empty file"}), 400

        img = Image.open(io.BytesIO(raw)).convert("RGB")
        m = get_model()
        results = m.predict(img, device=DEVICE, conf=CONFIDENCE,
                            imgsz=IMAGE_SIZE, verbose=False)
        r = results[0]

        objects, counts = [], {}
        if r.boxes is not None and len(r.boxes) > 0:
            cls = r.boxes.cls.detach().cpu().numpy().tolist()
            conf = r.boxes.conf.detach().cpu().numpy().tolist()
            for c, cf in sorted(zip(cls, conf), key=lambda x: x[1], reverse=True):
                label = CLASS_NAMES.get(int(c), f"class_{int(c)}")
                objects.append({"label": label, "confidence": float(cf)})
                counts[label] = counts.get(label, 0) + 1
        return jsonify({"objects": objects, "counts": counts})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
