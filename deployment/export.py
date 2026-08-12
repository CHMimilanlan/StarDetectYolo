from __future__ import annotations

import argparse

from utils.common import parse_device, project_path


def main() -> None:
    parser = argparse.ArgumentParser(description="把 best.pt 导出为 ONNX、OpenVINO、TensorRT 等格式。")
    parser.add_argument("--model", default="runs/train/yolov8n_stars/weights/best.pt")
    parser.add_argument("--format", default="onnx", choices=["onnx", "openvino", "engine", "torchscript"])
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--simplify", action="store_true")
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(str(project_path(args.model)))
    exported = model.export(
        format=args.format, imgsz=args.imgsz, device=parse_device(args.device),
        half=args.half, dynamic=args.dynamic, simplify=args.simplify,
    )
    print(f"导出完成: {exported}")


if __name__ == "__main__":
    main()
