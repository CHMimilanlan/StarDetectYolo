from __future__ import annotations

import argparse
import math
import os
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from utils.common import IMAGE_EXTENSIONS, find_images, labels_to_text, project_path


@dataclass(frozen=True)
class StarExtractionRuntime:
    load_image: Callable[..., Any]
    extract: Callable[..., Any]
    extract_params: dict[str, Any]
    stretch_params: dict[str, Any]


def load_star_extraction_runtime(
    raspa_root: Path,
    config_path: Path,
) -> StarExtractionRuntime:
    """延迟导入 RASPAstroStacker，使 --help 不依赖其完整运行环境。"""
    if not (raspa_root / "StarExtraction" / "extract.py").is_file():
        raise FileNotFoundError(f"无效的 RASPAstroStacker 目录: {raspa_root}")
    if not config_path.is_file():
        raise FileNotFoundError(f"找不到 RASPAstroStacker 配置: {config_path}")

    package_parent = str(raspa_root.parent)
    if package_parent not in sys.path:
        sys.path.insert(0, package_parent)
    # RASPAstroStacker.base.LoadYAML 会优先读取该变量。仅影响当前进程。
    os.environ["RASPAstroStacker_CONFIG"] = str(config_path)

    try:
        from RASPAstroStacker.StarExtraction.extract import ExtractionAPI
        from RASPAstroStacker.StarExtraction.main import (
            LoadExtractParam,
            LoadStretchImage,
            LoadStretchParam,
        )
    except ImportError as exc:
        raise RuntimeError(
            "无法导入 StarExtraction API。请先安装 RASPAstroStacker 的依赖: "
            "python -m pip install -r ../RASPAstroStacker/requirements.txt"
        ) from exc

    extract_params = LoadExtractParam("solver")
    if extract_params.get("mode") != "solver":
        raise ValueError("配置中的 solver 模式未正确设置 extract_param.star_filter.mode")
    return StarExtractionRuntime(
        LoadStretchImage,
        ExtractionAPI,
        extract_params,
        LoadStretchParam(),
    )


def points_to_boxes(
    stars: Any,
    width: int,
    height: int,
    box_size: float,
    image_path: Path,
) -> list[tuple[int, float, float, float, float]]:
    if not isinstance(stars, (list, tuple)):
        raise TypeError(
            f"ExtractionAPI 对 {image_path.name} 返回了 {type(stars).__name__}，"
            "solver 模式应返回星点列表"
        )

    half = box_size / 2.0
    boxes: list[tuple[int, float, float, float, float]] = []
    for index, star in enumerate(stars):
        if not isinstance(star, dict) or "x" not in star or "y" not in star:
            raise ValueError(f"{image_path.name} 的第 {index} 个星点缺少 x/y: {star!r}")
        try:
            x, y = float(star["x"]), float(star["y"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{image_path.name} 的第 {index} 个星点坐标不是数字: {star!r}"
            ) from exc
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ValueError(f"{image_path.name} 的第 {index} 个星点坐标含 NaN/Inf")
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError(
                f"{image_path.name} 的第 {index} 个星点 ({x}, {y}) "
                f"超出 {width}x{height} 图像"
            )

        x1, y1 = max(0.0, x - half), max(0.0, y - half)
        x2, y2 = min(float(width), x + half), min(float(height), y + half)
        if x2 > x1 and y2 > y1:
            boxes.append((0, x1, y1, x2, y2))
    return boxes


def write_label_atomic(label_path: Path, text: str) -> None:
    label_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = label_path.with_suffix(label_path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(label_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="批量调用 RASPAstroStacker 星点检测 API，并生成单类别 YOLO 标签。"
    )
    parser.add_argument(
        "--images",
        default="data/raw/images",
        help="输入图像目录（递归扫描）或单张图像，默认: data/raw/images",
    )
    parser.add_argument(
        "--output-labels",
        default="data/raw/labels",
        help="YOLO 标签输出目录，默认: data/raw/labels",
    )
    parser.add_argument(
        "--raspa-root",
        default="../RASPAstroStacker",
        help="RASPAstroStacker 项目目录，默认: ../RASPAstroStacker",
    )
    parser.add_argument(
        "--raspa-config",
        default=None,
        help="RASPAstroStacker config.yaml；默认使用 <raspa-root>/config.yaml",
    )
    parser.add_argument(
        "--box-size",
        type=float,
        default=8.0,
        help="以检测中心为中心的正方形标签边长（原图像素），默认: 8",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="允许覆盖已经存在的同名标签",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="单张图处理失败后继续；失败图不会写入或覆盖标签",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="启用 StarExtraction 调试输出（速度较慢）",
    )
    parser.add_argument(
        "--debug-output",
        default="outputs/star_extraction_debug",
        help="调试图输出目录，默认: outputs/star_extraction_debug",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not math.isfinite(args.box_size) or args.box_size <= 0:
        parser.error("--box-size 必须是大于 0 的有限数")

    image_source = project_path(args.images)
    output_dir = project_path(args.output_labels)
    raspa_root = project_path(args.raspa_root)
    config_path = (
        project_path(args.raspa_config)
        if args.raspa_config
        else raspa_root / "config.yaml"
    )
    debug_output = project_path(args.debug_output)

    if image_source.is_file():
        images = [image_source] if image_source.suffix.lower() in IMAGE_EXTENSIONS else []
    elif image_source.is_dir():
        images = find_images(image_source)
    else:
        raise SystemExit(f"输入图像路径不存在: {image_source}")
    if not images:
        raise SystemExit(f"没有找到支持的图像: {image_source}")

    duplicate_stems = sorted(
        stem for stem, count in Counter(path.stem for path in images).items() if count > 1
    )
    if duplicate_stems:
        raise SystemExit(
            "图像 basename 必须唯一，否则会产生同名标签。重复示例: "
            + ", ".join(duplicate_stems[:5])
        )

    existing = [output_dir / f"{path.stem}.txt" for path in images]
    existing = [path for path in existing if path.exists()]
    if existing and not args.overwrite:
        raise SystemExit(
            f"发现 {len(existing)} 个已有标签（例如 {existing[0]}）；"
            "确认后使用 --overwrite"
        )

    runtime = load_star_extraction_runtime(raspa_root, config_path)
    succeeded = 0
    failed: list[tuple[Path, Exception]] = []
    object_count = 0

    for index, image_path in enumerate(images, start=1):
        print(f"[{index}/{len(images)}] 检测 {image_path}")
        try:
            image = runtime.load_image(
                image_path,
                runtime.stretch_params,
                do_debug=args.debug,
            )
            if image is None or not hasattr(image, "shape") or len(image.shape) < 2:
                raise ValueError("图像加载结果无效")
            height, width = image.shape[:2]
            stars = runtime.extract(
                image,
                img_name=image_path.stem,
                extract_params=runtime.extract_params,
                do_debug=args.debug,
                debug_tmp_path=str(debug_output),
                save_small_stars=False,
            )
            boxes = points_to_boxes(stars, width, height, args.box_size, image_path)
            label_path = output_dir / f"{image_path.stem}.txt"
            write_label_atomic(label_path, labels_to_text(boxes, width, height))
            succeeded += 1
            object_count += len(boxes)
            print(f"  写入 {label_path.name}: {len(boxes)} 个星点")
        except Exception as exc:
            failed.append((image_path, exc))
            print(f"  失败: {type(exc).__name__}: {exc}", file=sys.stderr)
            if not args.continue_on_error:
                raise SystemExit(
                    "批处理已停止；失败图未写入标签。修复问题后重试，"
                    "或使用 --continue-on-error 跳过失败图。"
                ) from exc

    print(
        f"完成: 成功 {succeeded}/{len(images)} 张，失败 {len(failed)} 张，"
        f"共 {object_count} 个星点；标签目录: {output_dir}"
    )
    if failed:
        print("失败文件:", file=sys.stderr)
        for path, exc in failed:
            print(f"  {path}: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
