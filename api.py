"""FastAPI service for object detection.

Run:
    uvicorn api:app --reload
Then open http://127.0.0.1:8000/docs

Endpoints:
    GET  /health           -> service status
    POST /predict          -> upload an image, get detections as JSON
    POST /predict/image    -> upload an image, get the annotated image back
    POST /predict/video    -> upload a video, get per-frame detections as JSON
"""
import os
import tempfile

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import Response
from ultralytics import YOLO

WEIGHTS = os.getenv("WEIGHTS", "yolov8n.pt")
model = YOLO(WEIGHTS)

app = FastAPI(title="Object Detection API", version="1.0.0")


def decode_image(data: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid or unsupported image file.")
    return img


def to_detections(result) -> list:
    detections = []
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cls = int(box.cls[0])
        detections.append({
            "class_id": cls,
            "class_name": result.names[cls],
            "confidence": round(float(box.conf[0]), 4),
            "bbox": {"x1": round(x1, 1), "y1": round(y1, 1),
                     "x2": round(x2, 1), "y2": round(y2, 1)},
        })
    return detections


@app.get("/health")
def health():
    return {"status": "ok", "weights": WEIGHTS}


@app.post("/predict")
def predict(file: UploadFile = File(...),
            conf: float = Query(0.4, ge=0.0, le=1.0)):
    img = decode_image(file.file.read())
    result = model(img, conf=conf, verbose=False)[0]
    detections = to_detections(result)
    return {
        "filename": file.filename,
        "width": img.shape[1],
        "height": img.shape[0],
        "count": len(detections),
        "detections": detections,
    }


@app.post("/predict/image")
def predict_image(file: UploadFile = File(...),
                  conf: float = Query(0.4, ge=0.0, le=1.0)):
    img = decode_image(file.file.read())
    result = model(img, conf=conf, verbose=False)[0]
    annotated = result.plot()  # boxes + labels + confidence drawn on the image
    ok, buf = cv2.imencode(".jpg", annotated)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode output image.")
    return Response(content=buf.tobytes(), media_type="image/jpeg")


@app.post("/predict/video")
def predict_video(file: UploadFile = File(...),
                  conf: float = Query(0.4, ge=0.0, le=1.0),
                  frame_stride: int = Query(5, ge=1, description="process every Nth frame"),
                  max_frames: int = Query(300, ge=1, le=5000)):
    suffix = os.path.splitext(file.filename or "")[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file.file.read())
        path = tmp.name

    try:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Could not read video file.")

        frames, index = [], 0
        while len(frames) < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if index % frame_stride == 0:
                result = model(frame, conf=conf, verbose=False)[0]
                frames.append({"frame": index, "detections": to_detections(result)})
            index += 1
        cap.release()
    finally:
        os.remove(path)

    return {
        "filename": file.filename,
        "frames_processed": len(frames),
        "frame_stride": frame_stride,
        "frames": frames,
    }
