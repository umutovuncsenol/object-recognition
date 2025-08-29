from pathlib import Path
from flask import Flask, request, jsonify
from ultralytics import YOLO
from flask_cors import CORS
from flask import make_response
import tempfile
import torch
import os
import numpy as np

# ---------- Config ----------
MODEL_PATH = Path("yolov8n.pt")
CONFIDENCE = 0.50
IMAGE_SIZE = 640

# Device selection
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

# Production settings
PORT = int(os.environ.get("PORT", 5000))
HOST = "0.0.0.0"  # Allow external connections
DEBUG = os.environ.get("DEBUG", "False").lower() == "true"
# ----------------------------

app = Flask(__name__)
CORS(app, origins=["*"])  # Configure CORS for production

# Load model once at startup
try:
    model = YOLO(MODEL_PATH.as_posix())
    CLASS_NAMES = model.names
    print(f"Model loaded successfully: {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    CLASS_NAMES = {}

# add at top
import numpy as np

# right after the model loads successfully
try:
    # run a dummy forward pass to compile/warm caches
    _ = model.predict(
        source=np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8),
        device=DEVICE, conf=CONFIDENCE, imgsz=IMAGE_SIZE, verbose=False
    )
    print("Warmup done")
except Exception as e:
    print("Warmup failed:", e)

@app.route("/", methods=["GET"])
def home():
    """Serve the main page"""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <title>DIP • Digital Image Processor</title>
      <style>
        html, body {
          height: 100%;
          margin: 0;
        }
        body {
          display: flex;
          flex-direction: column;
          font-family: Arial, sans-serif;
          color: #333;
          text-align: center;
        }

        header {
          background: #008080;
          color: white;
          padding: 20px;
          border-radius: 20px;
          width: 400px;
          align-self: center;
          box-shadow: 0 4px 8px rgba(0,0,0,0.1);
          margin-top: 16px;
        }

        section {
          flex: 1;
          padding: 20px;
          display: flex;
          flex-direction: column;
          gap: 20px;
          align-items: center;
        }

        .controls {
          display: flex;
          gap: 12px;
          align-items: center;
          justify-content: center;
          flex-wrap: wrap;
        }

        button {
          padding: 10px 15px;
          font-size: 16px;
          background: #20b2aa;
          color: white;
          border: none;
          border-radius: 20px;
          cursor: pointer;
          transition: background 0.2s ease;
        }
        button:hover { background: #006666; }

        .analysis {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 24px;
          width: min(1000px, 95%);
          align-items: stretch;
        }
        @media (max-width: 800px) {
          .analysis { grid-template-columns: 1fr; }
        }

        .panel {
          background: #ffffff;
          border: 1px solid #e5e5e5;
          border-radius: 14px;
          padding: 16px;
          text-align: left;
          min-height: 320px;
          box-shadow: 0 2px 6px rgba(0,0,0,0.04);
          display: flex;
          flex-direction: column;
        }
        .panel h3 {
          margin: 0 0 12px;
          font-size: 18px;
          color: #035e5e;
        }

        .preview-wrap {
            display: flex;
            align-items: center;
            justify-content: center;
            background: #fafafa;
            border: 1px dashed #cfd8dc;
            border-radius: 10px;
            min-height: 260px;
            padding: 12px;
            text-align: center;
        }
        img.preview {
          max-width: 100%;
          height: auto;
          border-radius: 8px;
          max-height: 500px;
          object-fit: contain;
        }

        .results {
            flex: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            background: #fafafa;
            border: 1px dashed #cfd8dc;
            border-radius: 10px;
            padding: 12px;
            min-height: 260px;
            white-space: pre-wrap;
            line-height: 1.4;
            text-align: center;
        }
        .muted { color: #666; }
        .error { color: #b00020; }

        .results ul {
          margin: 0;
          padding-left: 20px;
          text-align: left;
          list-style-type: disc;
          width: 100%;
          display: block;
        }

        footer {
          background: #e0e0e0;
          padding: 15px;
          color: #444;
          font-size: 14px;
        }
      </style>
    </head>
    <body>
      <header>
        <h1>DIP</h1>
        <h2>Digital Image Processor</h2>
        <p>Object Recognizer / Image Classifier</p>
      </header>

      <section>
        <div class="controls">
          <h2 style="margin:0;">Upload an image.</h2>
          <button onclick="openPicker()">Upload</button>
          <input id="fileInput" type="file" accept="image/*" style="display:none" />
        </div>

        <div class="analysis">
          <div class="panel">
            <h3>Uploaded Image</h3>
            <div id="previewWrap" class="preview-wrap">
              <span class="muted">No image uploaded yet.</span>
            </div>
          </div>

          <div class="panel">
            <h3>Detected Objects</h3>
            <div id="results" class="results">
              <span class="muted">Results will appear here after upload.</span>
            </div>
          </div>
        </div>
      </section>

      <footer>
        <p>XON Technology Inc.</p>
      </footer>

      <script>

      function openPicker() {
        document.getElementById('fileInput').click();
      }
  
          const API = "/api/detect";
            let lastReq = 0;
            let controller = null;
            
            const fileInput  = document.getElementById('fileInput');
            const previewWrap = document.getElementById('previewWrap');
            const resultsBox  = document.getElementById('results');
        
          // optional: wake server early so first upload is fast
          fetch("/api/ping").catch(()=>{});
        
          fileInput.addEventListener('change', () => {
            const file = fileInput.files[0];
            if (!file) return;
        
            // preview...
            const reader = new FileReader();
            reader.onload = () => {
              previewWrap.innerHTML = '';
              const img = document.createElement('img');
              img.src = reader.result;
              img.className = 'preview';
              previewWrap.appendChild(img);
            };
            reader.readAsDataURL(file);
        
            // build form
            const fd = new FormData();
            fd.append("image", file);
        
            // UI: detecting…
            resultsBox.innerHTML = "<span class='muted'>Detecting...</span>";
            resultsBox.style.alignItems = "center";
            resultsBox.style.justifyContent = "center";
        
            // cancel any in-flight request
            if (controller) controller.abort();
            controller = new AbortController();
            const reqId = ++lastReq;
        
            fetch(`${API}?t=${Date.now()}`, {
              method: "POST",
              body: fd,
              cache: "no-store",
              signal: controller.signal
            })
            .then(async r => {
              const text = await r.text();
              try { return { ok: r.ok, data: JSON.parse(text) }; }
              catch { throw new Error(`HTTP ${r.status} – ${text || 'non-JSON response'}`); }
            })
            .then(({ ok, data }) => {
              if (reqId !== lastReq) return;               // ignore stale responses
              resultsBox.innerHTML = "";
              if (!ok) {
                resultsBox.innerHTML = `<span class='error'>Error: ${JSON.stringify(data)}</span>`;
                return;
              }
              if (data.counts && Object.keys(data.counts).length) {
                const ul = document.createElement("ul");
                for (const [label, count] of Object.entries(data.counts)) {
                  const li = document.createElement("li");
                  li.textContent = `${label} × ${count}`;
                  ul.appendChild(li);
                }
                resultsBox.style.alignItems = "flex-start";
                resultsBox.style.justifyContent = "flex-start";
                resultsBox.appendChild(ul);
              } else {
                resultsBox.innerHTML = "<span class='muted'>No objects detected.</span>";
              }
            })
            .catch(err => {
              if (err.name === "AbortError" || reqId !== lastReq) return; // ignore
              resultsBox.innerHTML = `<span class='error'>Error: ${err.message}</span>`;
            });
          });
        </script>
    </body>
    </html>
    """

@app.route("/api/detect", methods=["POST"])
def detect():
    """
    multipart/form-data with field 'image'
    returns: {"objects":[{"label":str,"confidence":float}], "counts": {"label": int, ...}}
    """
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
        
    if "image" not in request.files:
        return jsonify({"error": "No file field 'image'"}), 400
    
    f = request.files["image"]
    if not f.filename:
        return jsonify({"error": "Empty filename"}), 400
    
    try:
        # Save to temp file for YOLO inference
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
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
            
            # Clean up temp file
            os.unlink(tmp.name)
            
        resp = make_response(jsonify({"objects": objects, "counts": counts}))
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        resp.headers["Pragma"] = "no-cache"
        return resp

@app.route("/api/ping")
def ping():
    return jsonify({
        "status": "ok", 
        "device": DEVICE, 
        "model": MODEL_PATH.name,
        "model_loaded": model is not None
    })

if __name__ == "__main__":
    print(f"Model: {MODEL_PATH} | device={DEVICE} | conf={CONFIDENCE} | imgsz={IMAGE_SIZE}")
    print(f"Starting server on {HOST}:{PORT}")
    app.run(host=HOST, port=PORT, debug=DEBUG)
