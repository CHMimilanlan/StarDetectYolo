from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".fits", ".fit", ".fts"}
FITS_EXTENSIONS = {".fits", ".fit", ".fts"}


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as file:
        value = yaml.safe_load(file) or {}
    if not isinstance(value, dict):
        raise ValueError(f"YAML 顶层必须是映射: {path}")
    return value


def dump_yaml(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, allow_unicode=True, sort_keys=False)


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def resolve_dataset_yaml(path: str | Path) -> Path:
    """把项目约定的相对数据根目录写成绝对路径，避免 Ultralytics 路径歧义。"""
    source = project_path(path)
    data = load_yaml(source)
    root = project_path(data.get("path", "."))
    data["path"] = root.as_posix()
    runtime = PROJECT_ROOT / ".runtime" / f"{source.stem}_resolved.yaml"
    dump_yaml(data, runtime)
    return runtime


def find_images(directory: str | Path) -> list[Path]:
    directory = Path(directory)
    return sorted(path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def _read_fits(path: Path) -> np.ndarray:
    import numpy as np

    try:
        from astropy.io import fits
    except ImportError as exc:
        raise RuntimeError("读取 FITS 需要安装 requirements-fits.txt") from exc
    data = fits.getdata(path)
    data = np.asarray(data)
    while data.ndim > 2:
        data = data[0]
    if data.ndim != 2:
        raise ValueError(f"只支持二维 FITS 图像: {path}, shape={data.shape}")
    return data


def read_image(path: str | Path, normalization: str = "auto", percentiles: tuple[float, float] = (0.5, 99.8)) -> np.ndarray:
    """读取常规/16-bit/FITS 图像并输出 YOLO 可用的 uint8 BGR。"""
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("图像处理依赖尚未安装，请先执行: pip install -r requirements.txt") from exc

    path = Path(path)
    if path.suffix.lower() in FITS_EXTENSIONS:
        image = _read_fits(path)
    else:
        raw = np.fromfile(path, dtype=np.uint8)
        image = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"无法解码图像: {path}")

    if normalization not in {"auto", "stretch", "preserve"}:
        raise ValueError(f"未知归一化模式: {normalization}")
    needs_stretch = normalization == "stretch" or (normalization == "auto" and image.dtype != np.uint8)
    if needs_stretch:
        finite = image[np.isfinite(image)]
        if finite.size == 0:
            raise ValueError(f"图像没有有限像素值: {path}")
        low, high = np.percentile(finite, percentiles)
        if not math.isfinite(float(low)) or not math.isfinite(float(high)) or high <= low:
            low, high = float(finite.min()), float(finite.max())
        if high <= low:
            image = np.zeros(image.shape, dtype=np.uint8)
        else:
            image = np.nan_to_num(image, nan=low, posinf=high, neginf=low)
            image = np.clip((image.astype(np.float32) - low) * (255.0 / (high - low)), 0, 255).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    elif image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"不支持的图像形状: {path}, shape={image.shape}")
    return np.ascontiguousarray(image)


def write_image(path: str | Path, image: np.ndarray) -> None:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("图像处理依赖尚未安装，请先执行: pip install -r requirements.txt") from exc

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower() or ".png"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise ValueError(f"无法编码图像: {path}")
    encoded.tofile(path)


def parse_yolo_labels(path: str | Path, width: int, height: int) -> list[tuple[int, float, float, float, float]]:
    """返回像素坐标 (class_id, x1, y1, x2, y2)。不存在的标签视为负样本。"""
    path = Path(path)
    if not path.exists():
        return []
    boxes: list[tuple[int, float, float, float, float]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{path}:{line_number} 应有 5 列，实际为 {len(parts)}")
        try:
            class_value, cx, cy, bw, bh = map(float, parts)
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number} 含非数字字段") from exc
        values = (class_value, cx, cy, bw, bh)
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"{path}:{line_number} 含 NaN/Inf")
        class_id = int(class_value)
        if class_value != class_id or class_id < 0:
            raise ValueError(f"{path}:{line_number} 类别 ID 必须是非负整数")
        if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
            raise ValueError(f"{path}:{line_number} 坐标必须是 0~1 的归一化 xywh")
        x1, y1 = (cx - bw / 2) * width, (cy - bh / 2) * height
        x2, y2 = (cx + bw / 2) * width, (cy + bh / 2) * height
        if x1 < -1e-4 or y1 < -1e-4 or x2 > width + 1e-4 or y2 > height + 1e-4:
            raise ValueError(f"{path}:{line_number} 边界框超出图像")
        boxes.append((class_id, max(0.0, x1), max(0.0, y1), min(float(width), x2), min(float(height), y2)))
    return boxes


def labels_to_text(boxes: Iterable[tuple[int, float, float, float, float]], width: int, height: int) -> str:
    lines: list[str] = []
    for class_id, x1, y1, x2, y2 in boxes:
        cx, cy = (x1 + x2) / (2 * width), (y1 + y2) / (2 * height)
        bw, bh = (x2 - x1) / width, (y2 - y1) / height
        lines.append(f"{class_id} {cx:.8f} {cy:.8f} {bw:.8f} {bh:.8f}")
    return "\n".join(lines) + ("\n" if lines else "")


def parse_device(value: str) -> int | str | list[int]:
    value = value.strip()
    if "," in value:
        return [int(item.strip()) for item in value.split(",")]
    return int(value) if value.lstrip("-").isdigit() else value
