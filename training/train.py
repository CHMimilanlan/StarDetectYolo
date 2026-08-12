from __future__ import annotations

import argparse
from pathlib import Path

from utils.common import load_yaml, parse_device, project_path, resolve_dataset_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="使用 YOLOv8n 训练星点检测模型。")
    parser.add_argument("--config", default="configs/training/yolov8n_star.yaml")
    parser.add_argument("--resume", metavar="LAST_PT", help="从 last.pt 恢复完整训练状态")
    parser.add_argument("--device", help="覆盖设备，例如 0、0,1、cpu")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch", type=float)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--name")
    parser.add_argument("--dry-run", action="store_true", help="只打印最终配置")
    args = parser.parse_args()

    config_path = project_path(args.config)
    config = load_yaml(config_path)
    train_args = dict(config.get("train", {}))
    for key in ("epochs", "imgsz", "name"):
        value = getattr(args, key)
        if value is not None:
            train_args[key] = value
    if args.batch is not None:
        train_args["batch"] = int(args.batch) if args.batch.is_integer() else args.batch
    if args.device is not None:
        train_args["device"] = parse_device(args.device)
    if "project" in train_args:
        train_args["project"] = str(project_path(train_args["project"]))

    if args.resume:
        model_path = project_path(args.resume)
        final_args = {"model": str(model_path), "resume": True}
    else:
        model_value = str(config.get("model", "yolov8n.pt"))
        model_candidate = Path(model_value).expanduser()
        model_path = str(project_path(model_candidate)) if model_candidate.suffix and project_path(model_candidate).exists() else model_value
        data_yaml = resolve_dataset_yaml(config.get("data", "configs/dataset/star_dataset.yaml"))
        train_args["data"] = str(data_yaml)
        final_args = {"model": model_path, **train_args}

    if args.dry_run:
        print("最终训练配置:")
        for key, value in final_args.items():
            print(f"  {key}: {value}")
        return

    from ultralytics import YOLO

    model = YOLO(final_args.pop("model"))
    model.train(**final_args)


if __name__ == "__main__":
    main()
