from pathlib import Path
import numpy as np
from ..ort.config import OrtConfig
from ..ort.session import create_session
from ..ort.fetch import run_tensors
from .._io import read_json, sha256


class ModelBundle:
    def __init__(self, spec, directory):
        self.spec, self.directory = spec, Path(directory)
        self.paths = {k: self.directory / v for k, v in spec.recipe["graphs"].items()}
        missing = [str(v) for v in self.paths.values() if not v.is_file()]
        if missing:
            raise FileNotFoundError(
                "Export " + spec.name + " first. Missing: " + ", ".join(missing)
            )
        manifest = self.directory / "model.json"
        manifest_data = read_json(manifest) if manifest.exists() else None
        if manifest_data is not None:
            recorded = manifest_data["files"]
            for key, path in self.paths.items():
                if recorded.get(path.name) != sha256(path):
                    raise ValueError("Model bundle changed: " + path.name)
        self.pre = (
            create_session(self.paths["preprocess"], OrtConfig.cpu())
            if "preprocess" in self.paths
            else None
        )
        self.post = (
            create_session(self.paths["postprocess"], OrtConfig.cpu())
            if "postprocess" in self.paths
            else None
        )
        self.mapping = (
            manifest_data.get(
                "network_input_map",
                spec.recipe.get("network_input_map", {"images": "images"}),
            )
            if manifest_data is not None
            else spec.recipe.get("network_input_map", {"images": "images"})
        )

    def prepare(self, value, metadata):
        name = self.spec.recipe["input"]["name"]
        context = {name: value}
        if self.pre:
            names = [v.name for v in self.pre.get_outputs()]
            context.update(
                zip(
                    names,
                    run_tensors(
                        self.pre,
                        names,
                        {v.name: context[v.name] for v in self.pre.get_inputs()},
                    ),
                )
            )
        result = {name: context[source] for name, source in self.mapping.items()}
        if any(x.dtype != np.float32 for x in result.values()):
            raise ValueError("Network requires FP32 features")
        return result, dict(metadata, context=context)

    def finish(self, outputs, metadata):
        if not self.post:
            return outputs
        context = {**metadata["context"], **outputs}
        names = [v.name for v in self.post.get_outputs()]
        return dict(
            zip(
                names,
                run_tensors(
                    self.post,
                    names,
                    {v.name: context[v.name] for v in self.post.get_inputs()},
                ),
            )
        )
