from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from dataset.prepare_dataset import tile_starts
from utils.common import find_images, parse_device, project_path, read_image, write_image


def nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> list[int]:
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        rest = order[1:]
        ix1, iy1 = np.maximum(x1[current], x1[rest]), np.maximum(y1[current], y1[rest])
        ix2, iy2 = np.minimum(x2[current], x2[rest]), np.minimum(y2[current], y2[rest])
        intersection = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
        union = areas[current] + areas[rest] - intersection
        iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
        order = rest[iou <= threshold]
    return keep


def main() -> None:
    parser = argparse.ArgumentParser(description="滑窗检测大幅天文图，并合并重叠区域结果。")
    parser.add_argument("--model", default="runs/train/yolov8n_stars/weights/best.pt")
    parser.add_argument("--source", required=True, help="图像或图像目录")
    parser.add_argument("--output", default="outputs/large")
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--overlap", type=float, default=0.2)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--tile-iou", type=float, default=0.5)
    parser.add_argument("--merge-iou", type=float, default=0.35)
    parser.add_argument("--max-det", type=int, default=3000)
    parser.add_argument("--device", default="0")
    parser.add_argument("--normalization", choices=["auto", "stretch", "preserve"], default="auto")
    args = parser.parse_args()
    if args.tile_size <= 0 or not 0 <= args.overlap < 1:
        parser.error("tile-size 必须大于 0，overlap 必须位于 [0, 1)")

    from ultralytics import YOLO

    source = project_path(args.source)
    images = [source] if source.is_file() else find_images(source)
    if not images:
        raise SystemExit(f"没有找到图像: {source}")
    output = project_path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(project_path(args.model)))
    csv_path = output / "detections.csv"
    total = 0
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["image", "x_center", "y_center", "width", "height", "confidence", "class_id"])
        for image_path in images:
            image = read_image(image_path, args.normalization)
            height, width = image.shape[:2]
            tiles, offsets = [], []
            all_boxes, all_scores, all_classes = [], [], []

            def flush() -> None:
                if not tiles:
                    return
                results = model.predict(
                    source=tiles, imgsz=args.tile_size, conf=args.conf, iou=args.tile_iou,
                    max_det=args.max_det, device=parse_device(args.device), verbose=False,
                )
                for result, (x0, y0) in zip(results, offsets):
                    if result.boxes is None:
                        continue
                    boxes = result.boxes.xyxy.cpu().numpy()
                    boxes[:, [0, 2]] += x0
                    boxes[:, [1, 3]] += y0
                    all_boxes.extend(boxes.tolist())
                    all_scores.extend(result.boxes.conf.cpu().numpy().tolist())
                    all_classes.extend(result.boxes.cls.cpu().numpy().astype(int).tolist())
                tiles.clear()
                offsets.clear()

            for y in tile_starts(height, args.tile_size, args.overlap):
                for x in tile_starts(width, args.tile_size, args.overlap):
                    tiles.append(image[y : min(y + args.tile_size, height), x : min(x + args.tile_size, width)])
                    offsets.append((x, y))
                    if len(tiles) >= args.batch:
                        flush()
            flush()
            boxes = np.asarray(all_boxes, dtype=np.float32).reshape(-1, 4)
            scores = np.asarray(all_scores, dtype=np.float32)
            classes = np.asarray(all_classes, dtype=np.int32)
            keep: list[int] = []
            for class_id in np.unique(classes):
                indices = np.flatnonzero(classes == class_id)
                keep.extend(indices[nms(boxes[indices], scores[indices], args.merge_iou)].tolist())
            keep.sort(key=lambda index: float(scores[index]), reverse=True)
            annotated = image.copy()
            for index in keep:
                x1, y1, x2, y2 = boxes[index]
                writer.writerow([image_path.name, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1, float(scores[index]), int(classes[index])])
                cv2.rectangle(annotated, (round(x1), round(y1)), (round(x2), round(y2)), (0, 255, 0), 1)
            write_image(output / f"{image_path.stem}_detected.png", annotated)
            total += len(keep)
            print(f"{image_path.name}: {len(keep)} 个目标")
    print(f"总计 {total} 个目标；CSV: {csv_path}")


if __name__ == "__main__":
    main()
