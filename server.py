import os, io, base64, cv2, numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

# -------- Config --------
# Use a writable directory on Render
PROJECT_DIR = os.environ.get("PROJECT_DIR", os.getcwd())  # e.g. /opt/render/project/src
MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(PROJECT_DIR, "models"))
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(MODEL_DIR, "best_v15m.pt"))
GDRIVE_FILE_ID = os.environ.get("GDRIVE_FILE_ID")  # <-- set in Render Env
CONF = float(os.environ.get("CONF", 0.14))
IOU = float(os.environ.get("IOU", 0.50))
IMGSZ = int(os.environ.get("IMGSZ", 896))
AUGMENT = os.environ.get("AUGMENT", "true").lower() == "true"


# -------- Ensure model on disk (download from Drive if missing) --------
os.makedirs(MODEL_DIR, exist_ok=True)
if not os.path.exists(MODEL_PATH):
    assert GDRIVE_FILE_ID, "GDRIVE_FILE_ID env var is required to auto-download the model."
    import gdown
    url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
    print(f"Downloading model from Drive ID {GDRIVE_FILE_ID} -> {MODEL_PATH}")
    gdown.download(url, MODEL_PATH, quiet=False, fuzzy=True)

# -------- Load model --------
print("Loading YOLO model:", MODEL_PATH)
model = YOLO(MODEL_PATH)
CLASS_NAMES = model.names

# -------- App --------
app = FastAPI(title="Scopa Detector API")

# CORS: allow all for now (you can restrict to your domain later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

@app.get("/health")
def health():
    return {
        "ok": True,
        "model_loaded": MODEL_PATH,
        "num_classes": len(CLASS_NAMES),
        "classes_preview": dict(list(CLASS_NAMES.items())[:5]),
        "conf": CONF, "iou": IOU, "imgsz": IMGSZ, "augment": AUGMENT
    }

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    return_image: bool = True,
    conf: float = CONF,
    iou: float = IOU,
    imgsz: int = IMGSZ,
    augment: bool = AUGMENT
):
    from PIL import Image, ImageOps
    img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    img = ImageOps.exif_transpose(img)

    res = model.predict(img, imgsz=imgsz, conf=conf, iou=iou, augment=augment, verbose=False)[0]
    dets = []
    if res.boxes is not None and len(res.boxes) > 0:
        xyxy = res.boxes.xyxy.cpu().numpy().tolist()
        confs = res.boxes.conf.cpu().numpy().tolist()
        clss  = res.boxes.cls.cpu().numpy().astype(int).tolist()
        for bb, cc, kk in zip(xyxy, confs, clss):
            dets.append({
                "bbox_xyxy": [float(x) for x in bb],
                "confidence": float(cc),
                "class_id": int(kk),
                "class_name": CLASS_NAMES[int(kk)]
            })

    payload = {"detections": dets}
    if return_image:
        annotated = res.plot()  # BGR
        ok, buf = cv2.imencode(".jpg", annotated)
        if ok:
            payload["annotated_image_base64"] = base64.b64encode(buf.tobytes()).decode("utf-8")
    return JSONResponse(payload)
