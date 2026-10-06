# Sources and attribution

The package code is licensed under Apache-2.0. This does not license independently acquired weights, datasets, upstream model source, or generated outputs. No upstream repository clone, trained checkpoint, or dataset is included here. Historical measurement records are not included in this distribution.

| Component | Source | Treatment |
|---|---|---|
| ORT numerical helpers | microsoft/onnxruntime, 1.20.2 | Runtime dependency under the MIT license. Private numerical API compatibility is checked by the development regression suite. |
| Torch/torchvision classifiers | pytorch/pytorch, pytorch/vision, 2.5.1/0.20.1 | Runtime/export dependencies. Original weight enums and checkpoint hashes are recorded in recipes. |
| YOLOv5 | ultralytics/yolov5, commit 915bbf294bb74c859f0b41f1c23bc395014ea679 | Upstream code is acquired only for explicit export. GPL-3.0 applies to that code. No prebuilt model asset is redistributed here. |
| RetinaFace | biubug6/Pytorch_Retinaface, commit b984b4b775b2c4dced95c1eadd195a5c7d32a60b | Upstream MIT code is acquired at export. The original MobileNet0.25 checkpoint checksum is checked. The preserved 640×640 protocol is described in the README. |
| SSD / DeepLab | tensorflow/models and original model-zoo checkpoints | Source code is licensed under Apache-2.0. Original archives and datasets retain their own terms. |
| WIDER AP and VOC IoU | Official evaluation definitions. NumPy implementation. | Dataset/evaluation-tool payload is not redistributed. |

Framework package license notices remain with installed packages. Before redistributing model weights or publishing prebuilt assets, check the corresponding source and dataset distribution terms. This repository does not establish blanket redistribution rights. No claim is made that ONNX format alone determines an upstream license.

Algorithm literature is listed in [README references](README.md#license).

The initial PoT implementation consulted the local `aimet-pot-quantization` project at revision `a25a88d6880d3392aa92643a5d769dc037fe40b9` (`core/quantize_pot.py`). This is an implementation source attribution, not a bundled component or public download dependency. The package uses ORT-derived base scales rather than that reference's `absmax / 128` convention or Concat scale equalization.
