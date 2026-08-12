from __future__ import annotations

import argparse
import csv
from pathlib import Path

from utils.common import parse_device, project_path


def main() -> None:
    parser = argparse.ArgumentParser(description="对普通尺寸的图像/目录/视频推理，并输出星点中心 CSV。")
    parser.add_argument("--model", default="runs/train/yolov8n_stars/weights/best.pt")
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", default="outputs/predict")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--max-det", type=int, default=3000)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    from ultralytics import YOLO

    output = project_path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    source_candidate = Path(args.source).expanduser()
    source = str(source_candidate.resolve()) if source_candidate.exists() else args.source
    model = YOLO(str(project_path(args.model)))
    results = model.predict(
        source=source, imgsz=args.imgsz, conf=args.conf, iou=args.iou,
        max_det=args.max_det, device=parse_device(args.device), save=True,
        project=str(output.parent), name=output.name, exist_ok=True, stream=True,
    )
    csv_path = output / "detections.csv"
    count = 0
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["image", "x_center", "y_center", "width", "height", "confidence", "class_id"])
        for result in results:
            if result.boxes is None:
                continue
            for xyxy, confidence, class_id in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy(), result.boxes.cls.cpu().numpy()):
                x1, y1, x2, y2 = map(float, xyxy)
                writer.writerow([Path(result.path).name, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1, float(confidence), int(class_id)])
                count += 1
    print(f"检测到 {count} 个目标；CSV: {csv_path}")


if __name__ == "__main__":
    main()
