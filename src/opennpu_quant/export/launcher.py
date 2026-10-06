import os, subprocess, sys
from .._io import read_json


def export(model, paths, *, weights=None, source=None, python=None):
    """Run the exporter with this interpreter unless python is supplied."""
    command = [
        python or sys.executable,
        "-m",
        "opennpu_quant.export.worker",
        model,
        "--destination",
        str(paths.models / model),
        "--weights-dir",
        str(paths.weights),
    ]
    if weights:
        command += ["--weights", str(weights)]
    if source:
        command += ["--source-dir", str(source)]
    from ..models.spec import ModelSpec

    if (
        ModelSpec.load(model).recipe["export"].get("fixtures")
        == "voc_validation_first3"
    ):
        root = paths.datasets.get("voc")
        manifest = paths.data / "voc/manifest.json"
        if root is None and manifest.exists():
            root = read_json(manifest)["root"]
        if root is None:
            raise ValueError(
                "Prepare VOC data before exporting DeepLab; its preserved parity fixtures are the first three VOC validation images"
            )
        command += ["--voc-root", str(root)]
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="",
        TF_CPP_MIN_LOG_LEVEL="2",
        YOLOv5_AUTOINSTALL="false",
        PYTHONDONTWRITEBYTECODE="1",
    )
    cache = paths.cache / "export"
    cache.mkdir(parents=True, exist_ok=True)
    env.update(
        MPLCONFIGDIR=str(cache / "matplotlib"),
        YOLOV5_CONFIG_DIR=str(cache / "yolo"),
        TORCH_HOME=str(paths.weights / "torch"),
    )
    subprocess.run(command, env=env, check=True)
    return read_json(paths.models / model / "model.json")
