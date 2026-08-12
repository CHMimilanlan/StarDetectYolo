from __future__ import annotations

import argparse

from utils.common import parse_device, project_path, resolve_dataset_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="在 val/test 集上评估训练后的模型。")
    parser.add_argument("--model", default="runs/train/yolov8n_stars/weights/best.pt")
    parser.add_argument("--data", default="configs/dataset/star_dataset.yaml")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.001, help="计算 PR 曲线时应保持较低")
    parser.add_argument("--iou", type=float, default=0.6)
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(str(project_path(args.model)))
    metrics = model.val(
        data=str(resolve_dataset_yaml(args.data)), split=args.split, imgsz=args.imgsz,
        batch=args.batch, device=parse_device(args.device), conf=args.conf, iou=args.iou,
        project=str(project_path("runs/val")), name=f"{args.split}_stars", plots=True,
    )
    print(f"mAP50: {metrics.box.map50:.6f}")
    print(f"mAP50-95: {metrics.box.map:.6f}")
    print(f"Precision: {metrics.box.mp:.6f}")
    print(f"Recall: {metrics.box.mr:.6f}")


if __name__ == "__main__":
    main()
