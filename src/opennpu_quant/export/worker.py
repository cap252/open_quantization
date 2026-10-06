from pathlib import Path
import argparse, tempfile
from .._io import atomic_json, sha256, now
from .._locking import RunLock
from ..models.spec import ModelSpec
from ..models.preprocess import preprocess
from .fixtures import synthetic_image
from .torch_models import checkpoint, checkout
from .finalize import finalize


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("--destination", required=True)
    p.add_argument("--weights-dir", required=True)
    p.add_argument("--weights")
    p.add_argument("--source-dir")
    p.add_argument("--voc-root")
    a = p.parse_args()
    recipe = ModelSpec.load(a.model).recipe
    destination = Path(a.destination)
    destination.mkdir(parents=True, exist_ok=True)
    with RunLock(destination / ".lock"):
        weight = checkpoint(recipe, a.weights_dir, a.weights)
        source = (
            checkout(recipe, a.weights_dir, a.source_dir)
            if "repo" in recipe["source"]
            else None
        )
        shape = recipe["input"]["shape"]
        size = shape[2] if recipe["input"]["layout"] == "NCHW" else shape[1]
        feeds = []
        for index in range(3):
            if recipe["export"].get("fixtures") == "voc_validation_first3":
                if not a.voc_root:
                    raise ValueError("VOC parity images required")
                from PIL import Image

                names = (
                    (Path(a.voc_root) / "ImageSets/Segmentation/val.txt")
                    .read_text()
                    .split()[:3]
                )
                with Image.open(
                    Path(a.voc_root) / "JPEGImages" / (names[index] + ".jpg")
                ) as opened:
                    image = opened.convert("RGB")
            else:
                w, h = (
                    (size + 83, size + 47)
                    if recipe["task"] == "classification"
                    else (size + 53, size + 21)
                )
                image = synthetic_image(index, w, h)
            if recipe["export"]["family"] == "tf_ssd":
                from PIL import Image
                import numpy as np

                value = np.asarray(
                    image.resize((size, size), Image.Resampling.BILINEAR),
                    dtype=np.uint8,
                )[None]
            else:
                value, _ = preprocess(
                    image, recipe["preprocess"], recipe["input"]["layout"]
                )
            feeds.append({"images": value})
        with tempfile.TemporaryDirectory(prefix="export_", dir=destination) as scratch:
            raw = Path(scratch) / "raw.onnx"
            if recipe["export"]["family"].startswith("tf_"):
                from .tensorflow_models import export_model

                references = export_model(recipe, weight, feeds, raw, scratch)
            else:
                from .torch_models import export_model

                references = export_model(recipe, weight, source, feeds, raw)
            result = finalize(raw, recipe, feeds, references, Path(scratch))
            for name in recipe["graphs"].values():
                (Path(scratch) / name).replace(destination / name)
        from ..envcheck import info

        atomic_json(
            destination / "model.json",
            dict(
                schema_version=1,
                name=recipe["name"],
                source=recipe["source"],
                checkpoint_sha256=sha256(weight),
                files={
                    name: sha256(destination / name)
                    for name in recipe["graphs"].values()
                },
                network_input_map=result["network_input_map"],
                network_outputs=result["network_outputs"],
                validation=result,
                environment=info(),
                created=now(),
            ),
        )


if __name__ == "__main__":
    main()
