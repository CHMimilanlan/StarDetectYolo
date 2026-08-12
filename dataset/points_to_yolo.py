from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from utils.common import find_images, labels_to_text, project_path, read_image


def main() -> None:
    parser = argparse.ArgumentParser(description="把星点中心 CSV 转成单类别 YOLO 检测框标签。")
    parser.add_argument("--csv", required=True, help="至少包含 image,x,y 三列；x/y 为像素坐标")
    parser.add_argument("--images", default="data/raw/images")
    parser.add_argument("--output-labels", default="data/raw/labels")
    parser.add_argument("--image-column", default="image")
    parser.add_argument("--x-column", default="x")
    parser.add_argument("--y-column", default="y")
    parser.add_argument("--box-size", type=float, default=8.0, help="以中心点为中心的正方形边长（像素）")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.box_size <= 0:
        parser.error("box-size 必须大于 0")

    csv_path, images_dir, output = project_path(args.csv), project_path(args.images), project_path(args.output_labels)
    images = find_images(images_dir)
    by_name = {path.name: path for path in images}
    by_stem = {path.stem: path for path in images}
    points: dict[Path, list[tuple[float, float]]] = defaultdict(list)
    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = {args.image_column, args.x_column, args.y_column}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise SystemExit(f"CSV 缺少列 {sorted(required)}；现有列: {reader.fieldnames}")
        for row_number, row in enumerate(reader, start=2):
            key = Path(row[args.image_column]).name
            image_path = by_name.get(key) or by_stem.get(Path(key).stem)
            if image_path is None:
                raise SystemExit(f"CSV 第 {row_number} 行找不到图像: {key}")
            try:
                points[image_path].append((float(row[args.x_column]), float(row[args.y_column])))
            except ValueError as exc:
                raise SystemExit(f"CSV 第 {row_number} 行坐标不是数字") from exc

    output.mkdir(parents=True, exist_ok=True)
    object_count = 0
    for image_path in images:
        label_path = output / f"{image_path.stem}.txt"
        if label_path.exists() and not args.overwrite:
            raise SystemExit(f"标签已存在: {label_path}；确认后使用 --overwrite")
        height, width = read_image(image_path).shape[:2]
        half = args.box_size / 2
        boxes = []
        for x, y in points.get(image_path, []):
            if not (0 <= x < width and 0 <= y < height):
                raise SystemExit(f"点 ({x}, {y}) 超出图像 {image_path.name} 的 {width}x{height} 范围")
            x1, y1, x2, y2 = max(0.0, x - half), max(0.0, y - half), min(float(width), x + half), min(float(height), y + half)
            boxes.append((0, x1, y1, x2, y2))
        label_path.write_text(labels_to_text(boxes, width, height), encoding="utf-8")
        object_count += len(boxes)
    print(f"已为 {len(images)} 张图像写入 {object_count} 个星点标签: {output}")


if __name__ == "__main__":
    main()
