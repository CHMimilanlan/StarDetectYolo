# 基于 YOLOv8n 的星点检测

本项目提供从星点标注、原图级数据划分、重叠切片、标签检查，到 YOLOv8n 训练、评估、普通推理、大图滑窗推理和模型导出的完整流程。代码已按职责划分为独立模块，默认只检测一个类别：`star`。

> 星点是典型的小目标/密集目标。不要把 4K 或更大的天文图直接缩放到 640 像素训练；星点会丢失。推荐先按 1024×1024 重叠切片，并确保训练与推理采用一致的 8/16-bit 归一化策略。

## 1. 工程流程

1. 收集图像，并确定“星点”的标注口径（是否包含饱和星、拖线、热像素等）。
2. 使用 CVAT、LabelImg 等导出 YOLO 检测框，或者用 `points_to_yolo.py` 将星点中心 CSV 转为固定大小的框。
3. 按**原始图像**划分 train/val/test，再切片，避免同一原图的相邻切片泄漏到不同集合。
4. 用 `check_dataset.py` 检查图片、标签、类别和过小框。
5. 从 `yolov8n.pt` 迁移学习，查看 loss、PR 曲线、混淆矩阵和实际漏检/误检样例。
6. 在独立 test 集评估，按应用需求选择置信度阈值。
7. 普通图使用 `predict.py`；大图使用 `infer_large.py` 滑窗推理并跨切片 NMS 去重。
8. 如需部署，再导出 ONNX/OpenVINO/TensorRT。

主要目录结构：

```text
StarDetectYolo/
├── configs/
│   ├── dataset/       # 数据集路径与类别配置
│   └── training/      # 模型与训练超参数
├── dataset/           # 标注转换、数据划分、切片和检查
├── training/          # 训练与断点恢复
├── evaluation/        # val/test 定量评测
├── inference/         # 普通图与高分辨率大图推理
├── deployment/        # ONNX/OpenVINO/TensorRT 导出
└── utils/             # 公共路径、图像、标签和 YAML 工具
```

所有命令都应在项目根目录执行，并使用 `python -m 包名.模块名`，这样模块导入与路径解析不依赖当前脚本位置。

## 2. 环境配置

建议：Python 3.10 或 3.11、NVIDIA GPU（建议至少 8 GB 显存）、较新的显卡驱动。

在 PowerShell 中执行：

```powershell
cd C:\Workman02\python\OtherProject\ImageStack\StarDetectYolo
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

如果要读取 FITS：

```powershell
pip install -r requirements-fits.txt
```

`pip install ultralytics` 会安装 PyTorch，但未必是与你 CUDA 环境最合适的构建。若 `python -c "import torch; print(torch.cuda.is_available())"` 输出 `False`，请先按 [PyTorch 官方安装页](https://pytorch.org/get-started/locally/) 安装匹配的 CUDA 版 PyTorch，再执行 requirements 安装。训练接口和数据格式可参考 [Ultralytics 训练文档](https://docs.ultralytics.com/modes/train/) 与 [检测数据格式](https://docs.ultralytics.com/datasets/detect/)。

Ultralytics 有 AGPL-3.0 与商业许可选项；商业闭源部署前请确认许可证是否符合你的使用方式。

## 3. 准备原始数据和标签

默认原始目录：

```text
data/raw/
├── images/
│   ├── frame_001.tif
│   └── frame_002.png
└── labels/
    ├── frame_001.txt
    └── frame_002.txt
```

每个标签行是：

```text
class_id x_center y_center width height
0 0.512500 0.306250 0.007812 0.007812
```

坐标和宽高都必须除以图像宽高，归一化到 0~1。无星点图像可以没有 txt，也可以使用空 txt；适量真实负样本有助于抑制热像素、噪点和宇宙射线误检。

如果已有像素坐标 CSV：

```csv
image,x,y
frame_001.tif,123.4,567.8
frame_001.tif,900.0,412.0
```

转换为 8 像素方框：

```powershell
python -m dataset.points_to_yolo --csv data/raw/stars.csv --box-size 8
```

框尺寸应覆盖星点的 PSF 主体，并在同一数据集中保持口径一致。通常可先尝试 6~12 像素；框太小（尤其小于约 2 像素）难以学习，框太大会增加相邻星重叠。

## 4. 划分和切片数据集

推荐命令：

```powershell
python -m dataset.prepare_dataset --tile-size 1024 --overlap 0.2 --keep-empty 0.15
python -m dataset.check_dataset --data configs/dataset/star_dataset.yaml
```

脚本会先按原图以 8:1:1 划分，再切片并更新框坐标，输出：

```text
data/processed/
├── images/{train,val,test}/
└── labels/{train,val,test}/
```

关键参数：

- `--tile-size 1024`：切片边长；设为 `0` 可不切片。
- `--overlap 0.2`：20% 重叠，减少边缘星点被截断。
- `--min-visible 0.5`：至少保留原框 50% 面积。
- `--keep-empty 0.15`：随机保留 15% 空切片，防止负样本数量压倒正样本。
- `--normalization auto`：8-bit 原样保留；16-bit/FITS 按 0.5%~99.8% 百分位拉伸到 8-bit。
- `--overwrite`：明确允许重建非空的 `data/processed`。这会删除旧的处理后数据，原始数据不受影响。

若不同观测使用了不同曝光、增益或滤镜，应保证三个集合均有代表性。若需要评估跨设备泛化，最好按“观测批次/望远镜”分组划分，而不是随机按单帧划分；此时请手动组织 processed 目录。

## 5. 训练配置

先编辑 [configs/training/yolov8n_star.yaml](configs/training/yolov8n_star.yaml)：

- `device: 0`：第一块 NVIDIA GPU；无 GPU 改为 `cpu`（会很慢），多卡可用命令行 `--device 0,1`。
- `batch: -1`：让 Ultralytics 自动估计单卡 batch；显存不稳定时改成固定值，如 4 或 8。
- `imgsz: 1024`：应与切片边长一致，并最好是 32 的倍数。
- `epochs: 200`、`patience: 50`：作为基线，可根据验证集收敛情况调整。
- `workers: 4`：Windows 若 DataLoader 异常可改为 0。
- `project` / `name`：决定结果目录。

检查最终生效配置但不训练：

```powershell
python -m training.train --dry-run
```

开始训练：

```powershell
python -m training.train
```

首次运行会下载 `yolov8n.pt`。也可临时覆盖参数：

```powershell
python -m training.train --device 0 --epochs 300 --batch 4 --imgsz 1024 --name experiment_02
```

从中断点完整恢复：

```powershell
python -m training.train --resume runs/train/yolov8n_stars/weights/last.pt
```

最佳模型默认位于 `runs/train/yolov8n_stars/weights/best.pt`。同时查看 `results.png`、`PR_curve.png`、`confusion_matrix.png` 和验证集预测图。密集星场中 `max_det` 太小会截断结果，本项目推理默认设为 3000。

## 6. 评估和推理

在独立 test 集评估：

```powershell
python -m evaluation.evaluate --split test --model runs/train/yolov8n_stars/weights/best.pt
```

普通图像或目录：

```powershell
python -m inference.predict --source data/example --conf 0.15
```

高分辨率大图（推荐）：

```powershell
python -m inference.infer_large --source data/example/large.tif --tile-size 1024 --overlap 0.2 --conf 0.15
```

结果图和 `detections.csv` 位于 `outputs/predict` 或 `outputs/large`。CSV 中的 `x_center,y_center` 是原图像素坐标。训练阶段若对 16-bit 图使用 `auto` 拉伸，大图推理也要保持相同的 `--normalization auto`。

阈值选择不要只看 mAP：若任务更怕漏星，降低 `--conf`；若热像素误检成本更高，提高它。应在验证集上选择阈值，最后只在 test 集报告一次结果。

## 7. 导出

ONNX：

```powershell
python -m deployment.export --format onnx --imgsz 1024 --simplify
```

OpenVINO（CPU）或 TensorRT（NVIDIA）：

```powershell
python -m deployment.export --format openvino --imgsz 1024
python -m deployment.export --format engine --imgsz 1024 --device 0 --half
```

不同格式需要额外依赖，Ultralytics 会提示安装。导出接口见 [官方 Export 文档](https://docs.ultralytics.com/modes/export/)。

## 8. 常见问题与调参顺序

- **CUDA out of memory**：先降低 `batch`，再把 `imgsz` 从 1024 降到 768；不要一开始就牺牲小目标分辨率。
- **漏掉暗星**：确认输入拉伸一致；增加暗星标注；适度降低推理 `conf`；再考虑提高 `imgsz` 或换 `yolov8s.pt`。
- **把噪点当星**：加入暗场、热像素、云层、拖线等负样本；提高 `conf`；检查训练/验证是否来自不同原图。
- **密集区域检测数固定在上限**：增大 `--max-det`；训练验证时也可在 `train` 配置加入合适的 `max_det`。
- **切片接缝重复检测**：增大 `inference.infer_large` 的 `--merge-iou` 会更积极去重；若极小框位置波动导致 IoU 很低，可降低该阈值。
- **框非常小**：优先增大切片/输入中的星点像素尺度或统一标注框尺寸。YOLO 是框检测器；若最终只关心亚像素质心，可在 YOLO 粗检测后对每个 ROI 再做 PSF/高斯拟合。

建议一次只改变一组因素，并记录数据版本、随机种子、切片参数、归一化方式和权重文件。数据质量与划分方式通常比盲目增加 epochs 更重要。
