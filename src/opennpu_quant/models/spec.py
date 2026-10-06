from dataclasses import dataclass
from pathlib import Path
from importlib.resources import files
from .._io import read_yaml


@dataclass(frozen=True)
class ModelSpec:
    recipe: dict

    def __post_init__(self):
        validate_recipe_policy(self.recipe)

    @property
    def name(self):
        return self.recipe["name"]

    @classmethod
    def load(cls, name):
        path = Path(name)
        if path.is_file():
            data = read_yaml(path)
        else:
            if "/" in str(name) or ".." in str(name):
                raise ValueError("Unknown model: " + str(name))
            data = read_yaml(
                files("opennpu_quant.resources.models").joinpath(str(name) + ".yaml")
            )
        required = {
            "name",
            "title",
            "task",
            "dataset",
            "metric",
            "input",
            "graphs",
            "preprocess",
            "decode",
            "source",
            "export",
        }
        allowed = required | {
            "outputs",
            "network_outputs",
            "network_input_map",
            "metric_params",
        }
        if not isinstance(data, dict) or set(data) - allowed or required - set(data):
            raise ValueError("Invalid model recipe keys")
        if (
            set(data["graphs"]) - {"network", "preprocess", "postprocess"}
            or "network" not in data["graphs"]
        ):
            raise ValueError("Expected a network and optional host graphs")
        for value in data["graphs"].values():
            if Path(value).is_absolute() or ".." in Path(value).parts:
                raise ValueError("Unsafe graph path")
        return cls(data)


def model_names():
    return sorted(
        p.name[:-5]
        for p in files("opennpu_quant.resources.models").iterdir()
        if p.name.endswith(".yaml")
    )


def validate_framework_policy(recipe):
    policy = recipe.get("export", {}).get("framework_parity", {})
    if not isinstance(policy, dict):
        raise ValueError("framework_parity must be a mapping")
    for name, expected in (("rtol", 1e-4), ("atol", 1e-5)):
        if policy.get(name, expected) != expected:
            raise ValueError(
                "Frozen framework parity requires " + name + "=" + str(expected)
            )
    if (
        recipe.get("task") == "classification"
        and policy.get("require_top5_order", True) is not True
    ):
        raise ValueError("Classification framework parity requires Top5 order")


def validate_recipe_policy(recipe):
    """Reject edits to descriptive fields whose runtime behavior is fixed."""
    validate_framework_policy(recipe)
    # Explicit network boundaries must agree across recipe sections.
    split = recipe.get("export", {}).get("split", {})
    if (
        recipe.get("export", {}).get("family") != "tf_ssd"
        and "network_outputs" in recipe
        and "network_outputs" in split
        and recipe["network_outputs"] != split["network_outputs"]
    ):
        raise ValueError(
            "Conflicting network_outputs; use export.split.network_outputs"
        )
    pre = recipe.get("preprocess", {})
    decode = recipe.get("decode", {})
    inputs = recipe.get("input", {})

    def fixed(settings, key, expected, label):
        if key in settings and (
            settings[key] != expected
            or (isinstance(expected, bool) and type(settings[key]) is not bool)
        ):
            raise ValueError("Unsupported fixed recipe policy: " + label + "." + key)

    kind = pre.get("kind")
    if kind in (
        "center_crop",
        "letterbox",
        "resize_uint8",
        "resize_bgr_mean",
        "longer_side_pad",
    ):
        fixed(
            inputs, "dtype", "uint8" if kind == "resize_uint8" else "float32", "input"
        )
        if kind != "longer_side_pad":
            fixed(
                inputs, "layout", "NHWC" if kind == "resize_uint8" else "NCHW", "input"
            )
        elif inputs.get("layout", "NCHW") not in ("NCHW", "NHWC"):
            raise ValueError("Unsupported input layout")
    if kind in ("resize_uint8", "resize_bgr_mean"):
        fixed(pre, "interpolation", "cv2_linear", "preprocess")
    if kind == "longer_side_pad":
        fixed(pre, "pad", "zeros_bottom_right", "preprocess")
    if decode.get("kind") == "tf_od_api":
        fixed(decode, "boxes_to", "clip_xywh", "decode")
    if decode.get("kind") == "retinaface":
        fixed(decode.get("priors", {}), "clip", False, "decode.priors")
        fixed(decode, "rescale", "per_axis_clip", "decode")
        fixed(decode, "record", "wider_int_xywh", "decode")
    if decode.get("kind") == "yolo":
        for key, value in (
            ("score", "obj_x_cls"),
            ("boxes", "unletterbox_clip_xywh"),
            ("category_ids", "coco_sorted"),
        ):
            fixed(decode, key, value, "decode")
        for key, value in (
            ("impl", "cv2.dnn.NMSBoxes"),
            ("per_class", True),
            ("score_threshold", 0.0),
        ):
            fixed(decode.get("nms", {}), key, value, "decode.nms")
    params = recipe.get("metric_params", {})
    expected = {
        "segmentation_labels": {"classes": 21, "ignore_label": 255},
        "retinaface": {"iou": 0.5, "thresholds": 1000, "setting": "medium"},
    }.get(decode.get("kind"))
    if expected is not None:
        if not isinstance(params, dict) or set(params) - set(expected):
            raise ValueError("Unsupported metric_params")
        for key, value in expected.items():
            fixed(params, key, value, "metric_params")
    if recipe.get("export", {}).get("family") == "tf_deeplab":
        for key in ("graph_input", "graph_output"):
            value = recipe["export"].get(key)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError("Invalid DeepLab graph tensor name: " + key)
