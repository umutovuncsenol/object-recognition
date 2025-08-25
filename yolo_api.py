from pathlib import Path
from flask import Flask, request, jsonify
from ultralytics import YOLO
from flask_cors import CORS
import tempfile
import torch

# ---------- Config ----------
MODEL_PATH = Path("yolov8n.pt")
CONFIDENCE = 0.50
IMAGE_SIZE = 640

if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"
# ----------------------------

app = Flask(__name__)
CORS(app)  # allow calls from your separate index.html

# Load once at startup
model = YOLO(MODEL_PATH.as_posix())
CLASS_NAMES = model.names  # dict: id -> label

@app.route("/api/detect", methods=["POST"])
def detect():
    """
    multipart/form-data with field 'image'
    returns: {"objects":[{"label":str,"confidence":float}], "counts": {"label": int, ...}}
    """
    if "image" not in request.files:
        return jsonify({"error": "No file field 'image'"}), 400

    f = request.files["image"]
    if not f.filename:
        return jsonify({"error": "Empty filename"}), 400

    # Save to temp file for YOLO inference
    with tempfile.NamedTemporaryFile(suffix=".jpg") as tmp:
        f.save(tmp.name)

        # Run model
        results = model(tmp.name, device=DEVICE, conf=CONFIDENCE, imgsz=IMAGE_SIZE, verbose=False)
        r = results[0]

        objects = []
        counts = {}
        if r.boxes is not None and len(r.boxes) > 0:
            cls = r.boxes.cls.detach().cpu().numpy().tolist()
            conf = r.boxes.conf.detach().cpu().numpy().tolist()

            # sort by confidence desc
            for c, cf in sorted(zip(cls, conf), key=lambda x: x[1], reverse=True):
                label = CLASS_NAMES.get(int(c), f"class_{int(c)}")
                objects.append({"label": label, "confidence": float(cf)})
                counts[label] = counts.get(label, 0) + 1

        return jsonify({"objects": objects, "counts": counts})

@app.route("/api/ping")
def ping():
    return jsonify({"status": "ok", "device": DEVICE, "model": MODEL_PATH.name})

if __name__ == "__main__":
    print(f"Model: {MODEL_PATH} | device={DEVICE} | conf={CONFIDENCE} | imgsz={IMAGE_SIZE}")
    app.run(host="127.0.0.1", port=5000, debug=True)
