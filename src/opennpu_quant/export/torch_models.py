from pathlib import Path
import subprocess, sys
import numpy as np
from .._io import sha256
from ..fetch import download


def checkpoint(recipe, directory, explicit=None):
    source = recipe["source"]
    if explicit:
        path = Path(explicit).resolve()
        if sha256(path) != source["sha256"]:
            raise ValueError("Original checkpoint checksum mismatch")
        return path
    filename = source.get("file") or Path(source["url"].split("?")[0]).name
    return download(
        source["url"],
        Path(directory) / recipe["name"] / filename,
        sha256_expected=source["sha256"],
    )


def checkout(recipe, directory, explicit=None):
    source = recipe["source"]
    commit = source["commit"]
    target = (
        Path(explicit).resolve()
        if explicit
        else Path(directory) / "sources" / recipe["name"] / commit
    )
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--no-checkout", source["repo"], str(target)], check=True
        )
        subprocess.run(
            ["git", "-C", str(target), "checkout", "--detach", commit], check=True
        )
    current = subprocess.check_output(
        ["git", "-C", str(target), "rev-parse", "HEAD"], text=True
    ).strip()
    if current != commit:
        raise ValueError("Upstream source commit mismatch")
    changed = subprocess.check_output(
        ["git", "-C", str(target), "diff", "--name-only", "HEAD"], text=True
    ).strip()
    if changed:
        raise ValueError("Upstream tracked source has local changes")
    sys.path.insert(0, str(target))
    return target


def build(recipe, weight, source):
    import torch

    torch.set_num_threads(1)
    family = recipe["export"]["family"]
    transform = None
    if family == "torchvision":
        import torchvision

        name = recipe["source"]["builder"]
        kwargs = recipe["export"].get("kwargs", {})
        model = torchvision.models.get_model(name, weights=None, **kwargs)
        state = torch.load(weight, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        enum = torchvision.models.get_model_weights(name)
        transform = getattr(enum, recipe["source"]["weights"]).transforms()
    elif family == "retinaface":
        from models.retinaface import RetinaFace
        from data import cfg_mnet

        cfg = dict(cfg_mnet, pretrain=False)
        model = RetinaFace(cfg=cfg, phase="test")
        state = torch.load(weight, map_location="cpu", weights_only=True)
        if "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(
            {k.removeprefix("module."): v for k, v in state.items()}, strict=True
        )
    elif family == "yolov5":
        from models.experimental import attempt_load

        # Official checkpoint contains framework objects. Its pinned SHA256 is verified first.
        base = attempt_load(
            str(weight), device=torch.device("cpu"), inplace=False, fuse=True
        )

        class First(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, x):
                value = self.model(x)
                return value[0] if isinstance(value, (tuple, list)) else value

        model = First(base)
    else:
        raise ValueError("Unknown Torch exporter")
    return model.float().eval(), transform


def export_model(recipe, weight, source, feeds, target):
    import torch
    from .fixtures import synthetic_image
    from ..models.preprocess import preprocess

    model, transform = build(recipe, weight, source)
    if transform:
        crop = recipe["preprocess"]["crop"]
        for i, (w, h) in enumerate([(crop + 83, crop + 47), (321, 479), (479, 321)]):
            image = synthetic_image(i, w, h)
            actual, _ = preprocess(image, recipe["preprocess"])
            expected = transform(image).unsqueeze(0).numpy()
            np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)
    names = {
        "classification": ["logits"],
        "detection": ["predictions"],
        "face": ["loc", "conf", "landmarks"],
    }[recipe["task"]]
    references = []
    with torch.inference_mode():
        for feed in feeds:
            output = model(torch.from_numpy(feed["images"]))
            values = list(output) if isinstance(output, (tuple, list)) else [output]
            references.append({n: v.detach().numpy() for n, v in zip(names, values)})
        torch.onnx.export(
            model,
            torch.from_numpy(feeds[0]["images"]),
            str(target),
            opset_version=recipe["export"]["opset"],
            input_names=["images"],
            output_names=names,
            dynamo=False,
            do_constant_folding=True,
            training=torch.onnx.TrainingMode.EVAL,
        )
    return references
