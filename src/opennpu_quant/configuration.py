from pathlib import Path
from importlib.resources import files
import math, re
from ._io import read_yaml
from .quant.scheme import Scheme
from .quant.config import AdaroundConfig
from .quant.adaround_cache import ActivationCacheConfig


def _mapping(value, path, *, allowed=None, required=()):
    if not isinstance(value, dict):
        raise ValueError(path + " must be a mapping")
    missing = set(required) - value.keys()
    unknown = value.keys() - allowed if allowed is not None else set()
    if missing or unknown:
        kind, keys = ("Missing", missing) if missing else ("Unknown", unknown)
        raise ValueError(
            kind
            + " setting(s): "
            + ", ".join(f"{path}.{k}" for k in sorted(keys, key=str))
        )
    return value


def _device(value, path, *, allow_run=False):
    if not isinstance(value, str) or not (
        value == "cpu"
        or allow_run
        and value == "run"
        or value.startswith("cuda:")
        and value[5:].isdigit()
    ):
        choices = "run, cpu or cuda:N" if allow_run else "cpu or cuda:N"
        raise ValueError(f"{path} must be {choices}")


def load_config(name="core10"):
    path = Path(name)
    if not path.is_file():
        if str(name) not in ("core10", "quick"):
            raise ValueError("Unknown experiment: " + str(name))
        path = files("opennpu_quant.resources.experiments").joinpath(
            str(name) + ".yaml"
        )
    value = read_yaml(path)
    validate(value)
    return value


def _integer(value, path, minimum):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{path} must be an integer >= {minimum}")


def _boolean(value, path):
    if type(value) is not bool:
        raise ValueError(f"{path} must be boolean (true or false)")


def _number(value, path):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{path} must be a finite number")


def validate(value, *, models_dir=None):
    from .results import report_policy

    allowed = {
        "name",
        "models",
        "runtime",
        "calibration",
        "schemes",
        "conditions",
        "percentiles",
        "adaround",
        "activation_cache",
        "evaluation",
        "output",
        "report",
    }
    _mapping(
        value, "experiment", allowed=allowed, required=allowed - {"activation_cache"}
    )
    if not isinstance(value["models"], (list, tuple)):
        raise ValueError("models must be a list of model names")
    _mapping(value["conditions"], "conditions")
    if not value["conditions"]:
        raise ValueError("conditions must contain at least one condition")
    _mapping(value["schemes"], "schemes")
    _mapping(value["percentiles"], "percentiles")
    _mapping(value["adaround"], "adaround")
    for path, names in (
        ("name", [value["name"]]),
        ("models", value["models"]),
        ("conditions", value["conditions"]),
    ):
        if any(
            not isinstance(n, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", n)
            for n in names
        ):
            raise ValueError(
                path + " must use simple names (letters, digits, '_' or '-')"
            )
    if not value["models"] or len(value["models"]) != len(set(value["models"])):
        raise ValueError("Unique model names required")
    required = {
        "runtime": {"device", "intra_op_threads", "inter_op_threads"},
        "calibration": {"source", "device", "histogram_bins", "samples"},
        "evaluation": {"split", "limit"},
        "output": {"save_qdq_models", "save_predictions"},
    }
    for section, keys in {
        "runtime": {"device", "intra_op_threads", "inter_op_threads", "cuda"},
        "calibration": {"source", "device", "histogram_bins", "samples", "input_cache"},
        "evaluation": {"split", "limit"},
        "output": {"save_qdq_models", "save_predictions"},
        "report": {"primary_metric", "target_mean_recovery"},
        "activation_cache": {
            "enabled",
            "host_bytes",
            "device_bytes",
            "device_reserve_bytes",
        },
    }.items():
        _mapping(
            value.get(section, {}),
            section,
            allowed=keys,
            required=required.get(section, ()),
        )
    samples = _mapping(value["calibration"]["samples"], "calibration.samples")
    if not samples:
        raise ValueError("calibration.samples must contain dataset sample counts")
    for dataset, count in samples.items():
        _integer(count, f"calibration.samples.{dataset}", 1)
    for field in ("intra_op_threads", "inter_op_threads"):
        _integer(value["runtime"][field], "runtime." + field, 0)
    limit = value["evaluation"]["limit"]
    if limit is not None:
        _integer(limit, "evaluation.limit", 1)
    for field, flag in value["output"].items():
        _boolean(flag, "output." + field)
    for field, item in value.get("activation_cache", {}).items():
        if field == "enabled":
            _boolean(item, "activation_cache.enabled")
        else:
            _integer(item, "activation_cache." + field, 0)
    _device(value["runtime"]["device"], "runtime.device")
    _device(value["calibration"]["device"], "calibration.device", allow_run=True)
    _integer(value["calibration"]["histogram_bins"], "calibration.histogram_bins", 2)
    cache = value["calibration"].get("input_cache", {})
    _mapping(cache, "calibration.input_cache", allowed={"enabled", "max_bytes"})
    _boolean(cache.get("enabled", False), "calibration.input_cache.enabled")
    _integer(
        cache.get("max_bytes", 2 * 1024**3), "calibration.input_cache.max_bytes", 1
    )
    report_policy(value.get("report"))
    if value["calibration"]["source"] not in ("compute", "reference"):
        raise ValueError("Calibration source must be compute or reference")
    if value["evaluation"]["split"] != "validation":
        raise ValueError("Expected validation evaluation split")
    cuda = _mapping(
        value["runtime"].get("cuda", {}),
        "runtime.cuda",
        allowed={
            "use_tf32",
            "cudnn_conv_algo_search",
            "cudnn_conv_use_max_workspace",
            "gpu_mem_limit",
        },
    )
    for field in ("use_tf32", "cudnn_conv_use_max_workspace"):
        _boolean(cuda.get(field, False), "runtime.cuda." + field)
    _integer(cuda.get("gpu_mem_limit", 6 * 1024**3), "runtime.cuda.gpu_mem_limit", 1)
    for field, expected in (
        ("use_tf32", False),
        ("cudnn_conv_use_max_workspace", False),
        ("cudnn_conv_algo_search", "HEURISTIC"),
    ):
        if cuda.get(field, expected) != expected:
            raise ValueError(
                f"runtime.cuda.{field} must be {expected!r}; reference CUDA numerical policy is fixed"
            )
    from .models.spec import ModelSpec, model_names

    packaged = model_names()
    for model in value["models"]:
        local = (
            Path(models_dir) / model / "recipe.yaml" if models_dir is not None else None
        )
        source = local if local is not None and local.is_file() else model
        if models_dir is not None and (source != model or model in packaged):
            dataset = ModelSpec.load(source).recipe["dataset"]
            if dataset not in samples:
                raise ValueError(
                    f"Missing setting: calibration.samples.{dataset} (model '{model}')"
                )
        for family in value["schemes"]:
            scheme_for(value, model, family)
    for name, row in value["conditions"].items():
        _mapping(row, "conditions." + name, allowed={"scheme", "adaround"})
        if name == "fp32" and row:
            raise ValueError("conditions.fp32 must be an empty mapping: fp32: {}")
        if "adaround" in row:
            _boolean(row["adaround"], f"conditions.{name}.adaround")
        if row and (
            not isinstance(row.get("scheme"), str)
            or row["scheme"] not in value["schemes"]
        ):
            raise ValueError("Invalid conditions." + name + ".scheme")
        if not row and name != "fp32":
            raise ValueError("Only fp32 may omit a scheme")
    a = dict(value["adaround"])
    a.pop("data", None)
    for field in (
        "steps",
        "batch_size",
        "window_samples",
        "window_bytes",
        "window_steps",
        "seed",
    ):
        if field in a:
            _integer(a[field], "adaround." + field, 0 if field == "seed" else 1)
    for field in (
        "learning_rate",
        "regularization",
        "beta_start",
        "beta_end",
        "warmup_fraction",
        "atol",
        "rtol",
    ):
        if field in a:
            item = a[field]
            _number(item, "adaround." + field)
            positive = field in ("learning_rate", "beta_start", "beta_end")
            if (
                item < 0
                or positive
                and item == 0
                or field == "warmup_fraction"
                and item >= 1
            ):
                raise ValueError("Invalid adaround." + field + " range")
    if a.get("beta_start", 20) < a.get("beta_end", 2):
        raise ValueError("adaround.beta_start must be >= adaround.beta_end")
    try:
        AdaroundConfig(**a)
    except TypeError as error:
        raise ValueError("Invalid adaround setting: " + str(error)) from None
    if value["adaround"].get("data", "calibration") != "calibration":
        raise ValueError("AdaRound only uses calibration inputs")
    a = dict(value.get("activation_cache", {}))
    a.pop("enabled", None)
    try:
        ActivationCacheConfig(**a)
    except TypeError as error:
        raise ValueError("Invalid activation_cache setting: " + str(error)) from None


def scheme_for(config, model, family):
    fields = dict(_mapping(config["schemes"][family], f"schemes.{family}"))
    _boolean(
        fields.get("activation_symmetric", False),
        f"schemes.{family}.activation_symmetric",
    )
    bins = fields.get("quantized_bins", 128)
    _integer(bins, f"schemes.{family}.quantized_bins", 2)
    if bins > config["calibration"]["histogram_bins"]:
        raise ValueError(
            f"calibration.histogram_bins must be >= schemes.{family}.quantized_bins ({bins})"
        )
    if fields.get("percentile") == "per_model":
        path = f"percentiles.{model}"
        if model not in config["percentiles"]:
            raise ValueError("Missing setting: " + path)
        per_model = _mapping(config["percentiles"][model], path)
        if family not in per_model:
            raise ValueError("Missing setting: " + path + "." + family)
        percentile = per_model[family]
        _number(percentile, f"{path}.{family}")
        if not 0 < percentile <= 100:
            raise ValueError(f"{path}.{family} must be in (0, 100]")
        fields["percentile"] = percentile
    elif fields.get("percentile") is not None:
        _number(fields["percentile"], f"schemes.{family}.percentile")
        if not 0 < fields["percentile"] <= 100:
            raise ValueError(f"schemes.{family}.percentile must be in (0, 100]")
    fields["histogram_bins"] = config["calibration"]["histogram_bins"]
    try:
        return Scheme(**fields)
    except TypeError as error:
        raise ValueError(f"Invalid schemes.{family} setting: {error}") from None


def runtime(config, device=None):
    from .ort.config import OrtConfig

    policy = config["runtime"]
    device = device or policy["device"]
    _device(device, "runtime.device")
    opts = dict(
        intra_op_threads=policy["intra_op_threads"],
        inter_op_threads=policy["inter_op_threads"],
    )
    if device == "cpu":
        return OrtConfig.cpu(**opts)
    return OrtConfig.cuda(
        int(device[5:]),
        memory_limit=policy.get("cuda", {}).get("gpu_mem_limit", 6 * 1024**3),
        **opts,
    )
