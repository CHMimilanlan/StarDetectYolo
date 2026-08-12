from __future__ import annotations

import argparse
import random
import shutil
from collections import Counter
from pathlib import Path

from utils.common import PROJECT_ROOT, dump_yaml, find_images, labels_to_text, parse_yolo_labels, project_path, read_image, write_image


def tile_starts(length: int, tile_size: int, overlap: float) -> list[int]:
    if length <= tile_size:
        return [0]
    stride = max(1, int(round(tile_size * (1 - overlap))))
    starts = list(range(0, length - tile_size + 1, stride))
    final = length - tile_size
    if starts[-1] != final:
        starts.append(final)
    return starts


def clip_boxes(boxes, x0: int, y0: int, width: int, height: int, min_visible: float, min_box_pixels: float):
    output = []
    for class_id, x1, y1, x2, y2 in boxes:
        ix1, iy1 = max(x1, x0), max(y1, y0)
        ix2, iy2 = min(x2, x0 + width), min(y2, y0 + height)
        if ix2 <= ix1 or iy2 <= iy1:
            continue
        original_area = max(1e-9, (x2 - x1) * (y2 - y1))
        visible = (ix2 - ix1) * (iy2 - iy1) / original_area
        if visible < min_visible or ix2 - ix1 < min_box_pixels or iy2 - iy1 < min_box_pixels:
            continue
        output.append((class_id, ix1 - x0, iy1 - y0, ix2 - x0, iy2 - y0))
    return output


def split_sources(images: list[Path], train: float, val: float, seed: int) -> dict[str, list[Path]]:
    items = images.copy()
    random.Random(seed).shuffle(items)
    count = len(items)
    val_count = int(round(count * val))
    test_count = int(round(count * (1 - train - val)))
    if val > 0 and count >= 3:
        val_count = max(1, val_count)
    if 1 - train - val > 0 and count >= 3:
        test_count = max(1, test_count)
    if val_count + test_count >= count:
        test_count = max(0, count - val_count - 1)
    return {
        "train": items[val_count + test_count :],
        "val": items[:val_count],
        "test": items[val_count : val_count + test_count],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="校验原始 YOLO 标签，按原图划分数据集，并可切片大图。")
    parser.add_argument("--raw-images", default="data/raw/images")
    parser.add_argument("--raw-labels", default="data/raw/labels")
    parser.add_argument("--output", default="data/processed")
    parser.add_argument("--data-config", default="configs/dataset/star_dataset.yaml")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tile-size", type=int, default=1024, help="0 表示不切片")
    parser.add_argument("--overlap", type=float, default=0.2)
    parser.add_argument("--min-visible", type=float, default=0.5)
    parser.add_argument("--min-box-pixels", type=float, default=2.0)
    parser.add_argument("--keep-empty", type=float, default=0.15, help="保留空切片的概率")
    parser.add_argument("--normalization", choices=["auto", "stretch", "preserve"], default="auto")
    parser.add_argument("--low-percentile", type=float, default=0.5)
    parser.add_argument("--high-percentile", type=float, default=99.8)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not (0 < args.train_ratio < 1 and 0 <= args.val_ratio < 1 and args.train_ratio + args.val_ratio <= 1):
        parser.error("比例必须满足 0 < train < 1、0 <= val < 1、train + val <= 1")
    if not (0 <= args.overlap < 1 and 0 <= args.keep_empty <= 1 and 0 <= args.min_visible <= 1):
        parser.error("overlap、keep-empty、min-visible 必须位于有效范围")
    if not (0 <= args.low_percentile < args.high_percentile <= 100):
        parser.error("百分位范围无效")

    raw_images, raw_labels = project_path(args.raw_images), project_path(args.raw_labels)
    output, data_config = project_path(args.output), project_path(args.data_config)
    images = find_images(raw_images)
    if not images:
        raise SystemExit(f"没有找到原始图像: {raw_images}")
    duplicate_stems = [stem for stem, n in Counter(path.stem for path in images).items() if n > 1]
    if duplicate_stems:
        raise SystemExit(f"图像 basename 必须唯一，重复项示例: {duplicate_stems[:5]}")
    if output.exists() and any(output.iterdir()):
        if not args.overwrite:
            raise SystemExit(f"输出目录非空: {output}；确认后使用 --overwrite")
        shutil.rmtree(output)

    splits = split_sources(images, args.train_ratio, args.val_ratio, args.seed)
    rng = random.Random(args.seed)
    stats = Counter()
    for split, split_images in splits.items():
        for source in split_images:
            image = read_image(source, args.normalization, (args.low_percentile, args.high_percentile))
            height, width = image.shape[:2]
            boxes = parse_yolo_labels(raw_labels / f"{source.stem}.txt", width, height)
            if args.tile_size <= 0:
                targets = [(0, 0, width, height)]
            else:
                targets = [
                    (x, y, min(args.tile_size, width - x), min(args.tile_size, height - y))
                    for y in tile_starts(height, args.tile_size, args.overlap)
                    for x in tile_starts(width, args.tile_size, args.overlap)
                ]
            for x, y, tile_width, tile_height in targets:
                tile_boxes = clip_boxes(boxes, x, y, tile_width, tile_height, args.min_visible, args.min_box_pixels)
                if args.tile_size > 0 and not tile_boxes and rng.random() > args.keep_empty:
                    stats["discarded_empty_tiles"] += 1
                    continue
                name = source.stem if args.tile_size <= 0 else f"{source.stem}__x{x}_y{y}"
                write_image(output / "images" / split / f"{name}.png", image[y : y + tile_height, x : x + tile_width])
                label_path = output / "labels" / split / f"{name}.txt"
                label_path.parent.mkdir(parents=True, exist_ok=True)
                label_path.write_text(labels_to_text(tile_boxes, tile_width, tile_height), encoding="utf-8")
                stats[f"{split}_images"] += 1
                stats[f"{split}_objects"] += len(tile_boxes)

    try:
        dataset_root = output.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        dataset_root = output.as_posix()
    dump_yaml(
        {
            "path": dataset_root,
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": {0: "star"},
        },
        data_config,
    )
    print(f"数据集已生成: {output}")
    for key in sorted(stats):
        print(f"  {key}: {stats[key]}")
    print(f"数据配置: {data_config}")


if __name__ == "__main__":
    main()
