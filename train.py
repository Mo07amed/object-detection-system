"""Train YOLOv8n on a custom dataset.

Usage:
    python train.py --data data.yaml --epochs 50

After training, loss curves and metrics are saved automatically to
runs/train/ (results.png, confusion_matrix.png, weights/best.pt).
"""
import argparse

from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description="Train YOLOv8n")
    p.add_argument("--data", default="data.yaml", help="dataset config file")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    return p.parse_args()


def main():
    args = parse_args()
    model = YOLO("yolov8n.pt")  # pretrained Nano model: fast on modest hardware

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project="runs",
        name="train",
        # Data augmentation
        degrees=10.0,   # random rotation
        hsv_v=0.4,      # brightness / lighting variation
        fliplr=0.5,     # horizontal flip
        mosaic=1.0,
    )

    metrics = model.val()  # evaluate best weights on the validation set
    print(f"mAP@0.5:      {metrics.box.map50:.3f}")
    print(f"mAP@0.5:0.95: {metrics.box.map:.3f}")


if __name__ == "__main__":
    main()
