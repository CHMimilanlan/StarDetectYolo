from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from utils.common import find_images, load_yaml, parse_yolo_labels, project_path, read_image


def label_dir_for(image_dir: Path, dataset_root: Path) -> Path:
    relative = image_dir.relative_to(dataset_root)
    parts = list(relative.parts)
    if "images" not in parts:
        raise ValueError(f"数据路径应包含 images 目录层级: {image_dir}")
    parts[parts.index("images")] = "labels"
    return dataset_root.joinpath(*parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="检查数据文件、YOLO 标签格式和小目标尺寸。")
    parser.add_argument("--data", default="configs/dataset/star_dataset.yaml")
    parser.add_argument("--min-box-pixels", type=float, default=2.0)
    parser.add_argument("--normalization", choices=["auto", "stretch", "preserve"], default="auto")
    args = parser.parse_args()

    data_path = project_path(args.data)
    data = load_yaml(data_path)
    root = project_path(data.get("path", "."))
    names = data.get("names", {})
    class_count = len(names)
    errors: list[str] = []
    stats = Counter()
    for split in ("train", "val", "test"):
        if not data.get(split):
            continue
        image_dir = root / str(data[split])
        if not image_dir.exists():
            errors.append(f"{split} 图像目录不存在: {image_dir}")
            continue
        try:
            label_dir = label_dir_for(image_dir, root)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        images = find_images(image_dir)
        if not images:
            errors.append(f"{split} 没有图像: {image_dir}")
            continue
        for image_path in images:
            stats[f"{split}_images"] += 1
            try:
                image = read_image(image_path, args.normalization)
                height, width = image.shape[:2]
                label_path = label_dir / image_path.relative_to(image_dir).with_suffix(".txt")
                boxes = parse_yolo_labels(label_path, width, height)
                if not label_path.exists():
                    stats[f"{split}_missing_labels_as_negative"] += 1
                if not boxes:
                    stats[f"{split}_negative_images"] += 1
                for class_id, x1, y1, x2, y2 in boxes:
                    if class_id >= class_count:
                        errors.append(f"{label_path}: 类别 {class_id} 超出 names 范围")
                    if x2 - x1 < args.min_box_pixels or y2 - y1 < args.min_box_pixels:
                        stats[f"{split}_tiny_boxes"] += 1
                stats[f"{split}_objects"] += len(boxes)
            except Exception as exc:  # 汇总所有坏样本后一次性报告
                errors.append(f"{image_path}: {exc}")

    for key in sorted(stats):
        print(f"{key}: {stats[key]}")
    if errors:
        print(f"\n发现 {len(errors)} 个错误:")
        for error in errors[:30]:
            print(f"- {error}")
        if len(errors) > 30:
            print(f"- ... 另有 {len(errors) - 30} 个")
        raise SystemExit(1)
    print("\n数据集检查通过。")


if __name__ == "__main__":
    main()
