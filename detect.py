"""Real-time object detection from a camera, video file, or image.

Usage:
    python detect.py --source 0                       # webcam
    python detect.py --source video.mp4 --save out.mp4
    python detect.py --source photo.jpg
    python detect.py --weights runs/train/weights/best.pt --source 0

Press 'q' to quit.
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

FONT = cv2.FONT_HERSHEY_SIMPLEX
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    p = argparse.ArgumentParser(description="Real-time object detection")
    p.add_argument("--weights", default="yolov8n.pt", help="model weights (.pt)")
    p.add_argument("--source", default="0", help="camera index, video path, or image path")
    p.add_argument("--conf", type=float, default=0.4, help="confidence threshold")
    p.add_argument("--save", default=None, help="optional output path (e.g. out.mp4 / out.jpg)")
    return p.parse_args()


def color_for(class_id: int):
    """Stable color per class."""
    rng = np.random.RandomState(class_id)
    return tuple(int(c) for c in rng.randint(60, 255, 3))


def draw_detections(frame, result):
    """Draw bounding boxes, class names and confidence scores on the frame."""
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = f"{result.names[cls]} {conf:.0%}"
        color = color_for(cls)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, FONT, 0.55, 1)
        top = max(y1 - th - 8, 0)
        cv2.rectangle(frame, (x1, top), (x1 + tw + 6, top + th + 8), color, -1)
        cv2.putText(frame, label, (x1 + 3, top + th + 2), FONT, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def run_on_image(model, path, conf, save):
    frame = cv2.imread(path)
    if frame is None:
        raise SystemExit(f"Could not read image: {path}")
    result = model(frame, conf=conf, verbose=False)[0]
    out = draw_detections(frame, result)
    if save:
        cv2.imwrite(save, out)
        print(f"Saved: {save}")
    cv2.imshow("Object Detection", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_on_stream(model, source, conf, save):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {source}")

    writer = None
    prev = time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            result = model(frame, conf=conf, verbose=False)[0]
            frame = draw_detections(frame, result)

            now = time.time()
            fps = 1.0 / max(now - prev, 1e-6)
            prev = now
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 28), FONT, 0.8,
                        (0, 255, 0), 2, cv2.LINE_AA)

            if save and writer is None:
                h, w = frame.shape[:2]
                src_fps = cap.get(cv2.CAP_PROP_FPS) or 25
                writer = cv2.VideoWriter(save, cv2.VideoWriter_fourcc(*"mp4v"),
                                         src_fps, (w, h))
            if writer:
                writer.write(frame)

            cv2.imshow("Object Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        if writer:
            writer.release()
            print(f"Saved: {save}")
        cv2.destroyAllWindows()


def main():
    args = parse_args()
    model = YOLO(args.weights)

    if Path(args.source).suffix.lower() in IMAGE_EXTS:
        run_on_image(model, args.source, args.conf, args.save)
    else:
        source = int(args.source) if args.source.isdigit() else args.source
        run_on_stream(model, source, args.conf, args.save)


if __name__ == "__main__":
    main()
