from pathlib import Path
from .._io import read_json, object_hash, sha256
from ..models.contracts import EvaluationSample


def preprocessing_implementation(settings):
    """Return the supported feed implementation boundary, or unknown for plugins.

    Arbitrary Python callbacks can read helpers, mutable globals, files and remote
    state. Hashing just their function body/module cannot certify their feeds.
    """
    if ":" in settings["kind"]:
        return None
    import cv2
    import numpy
    import PIL
    from ..models import preprocess, bundle

    return dict(
        files={
            "preprocess": sha256(preprocess.__file__),
            "bundle": sha256(bundle.__file__),
            "dataset": sha256(__file__),
        },
        dependencies={
            "numpy": numpy.__version__,
            "pillow": PIL.__version__,
            "opencv": cv2.__version__,
        },
    )


class Dataset:
    def __init__(self, manifest, *, root=None):
        self.manifest = Path(manifest)
        self.info = read_json(self.manifest)
        self.root = Path(root or self.info["root"]).resolve()
        self.dataset = self.info["dataset"]
        self.annotations = {
            k: self.path(v) for k, v in self.info.get("annotations", {}).items()
        }
        self.annotation_identity = {k: sha256(v) for k, v in self.annotations.items()}

    def path(self, relative):
        p = (self.root / relative).resolve()
        if not p.is_relative_to(self.root):
            raise ValueError("Dataset path escapes root")
        return p

    def rows(self, split):
        rows = self.info[split]
        ids = [str(x["id"]) for x in rows]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate dataset IDs")
        return rows

    def identity(self, split, limit=None):
        # Root location is excluded. Detect content/annotation changes before reuse.
        return object_hash(
            [
                self.dataset,
                self.annotation_identity,
                [
                    (
                        r,
                        sha256(self.path(r["path"])),
                        sha256(self.path(r["target"]))
                        if isinstance(r.get("target"), str)
                        else None,
                    )
                    for r in self.rows(split)[:limit]
                ],
            ]
        )

    def samples(self, bundle, split="validation", limit=None):
        from PIL import Image
        from ..models.preprocess import preprocess
        import numpy as np

        recipe = bundle.spec.recipe
        for row in self.rows(split)[:limit]:
            with Image.open(self.path(row["path"])) as image:
                tensor, meta = preprocess(
                    image, recipe["preprocess"], recipe["input"]["layout"]
                )
            feeds, meta = bundle.prepare(tensor, meta)
            target = row.get("target")
            if recipe["task"] == "segmentation" and target is not None:
                with Image.open(self.path(target)) as image:
                    target = np.asarray(image).copy()
            meta["dataset_row"] = row
            yield EvaluationSample(str(row["id"]), feeds, target, meta)

    def feeds(self, bundle, count):
        if type(count) is not int or count < 1 or count > len(self.rows("calibration")):
            raise ValueError("Calibration sample count exceeds prepared split")

        def factory():
            for sample in self.samples(bundle, "calibration", count):
                yield sample.feeds

        return factory

    def cached_feeds(self, bundle, count, directory, *, max_bytes):
        from ..graph.model import load_model, model_identity
        from ..ort.config import OrtConfig
        from ..ort.session import environment
        from .feed_cache import prepare_feed_cache

        implementation = preprocessing_implementation(bundle.spec.recipe["preprocess"])
        if implementation is None:
            raise ValueError(
                "Input cache cannot identify custom preprocessing and its dependencies; "
                "disable calibration.input_cache for this recipe"
            )
        factory = self.feeds(bundle, count)
        source = self.identity("calibration", count)
        pre = bundle.paths.get("preprocess")
        pre_identity = model_identity(pre) if pre else None
        network = load_model(bundle.paths["network"])
        inputs = [value.SerializeToString().hex() for value in network.graph.input]
        del network
        identity = dict(
            schema_version=2,
            source=source,
            preprocessing=bundle.spec.recipe["preprocess"],
            input=bundle.spec.recipe["input"],
            input_map=bundle.mapping,
            network_inputs=inputs,
            preprocess_graph=pre_identity,
            implementation=implementation,
            environment={
                "cpu_preprocess": environment(OrtConfig.cpu()) if pre else None,
            },
        )

        settings_identity = object_hash(
            [
                bundle.spec.recipe["preprocess"],
                bundle.spec.recipe["input"],
                bundle.mapping,
            ]
        )

        def verify_source():
            if (
                object_hash(
                    [
                        bundle.spec.recipe["preprocess"],
                        bundle.spec.recipe["input"],
                        bundle.mapping,
                    ]
                )
                != settings_identity
            ):
                raise ValueError("Preprocessing settings/input mapping changed")
            if (
                preprocessing_implementation(bundle.spec.recipe["preprocess"])
                != implementation
            ):
                raise ValueError("Preprocessing implementation/dependencies changed")
            if self.identity("calibration", count) != source:
                raise ValueError("Calibration source/order changed")
            if pre and model_identity(pre) != pre_identity:
                raise ValueError("CPU preprocessing graph changed")

        return prepare_feed_cache(
            Path(directory) / object_hash(identity),
            factory,
            identity=identity,
            verify_source=verify_source,
            expected_samples=count,
            max_bytes=max_bytes,
        )
