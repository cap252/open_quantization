from contextlib import contextmanager, nullcontext
from dataclasses import replace, asdict
from pathlib import Path
import copy
import hashlib
import math
import os
import time
import numpy as np
import onnx
from onnx import helper, numpy_helper
from opennpu_quant._io import object_hash, atomic_json, read_json, sha256, now
from opennpu_quant.graph.model import load_model, model_identity
from opennpu_quant.ort.session import create_session, environment
from .calibration import feed_digest
from .config import AdaroundConfig
from .parameters import parameter_identity, verify_parameters
from .adaround_ops import attributes, operation, channel_axis


def frozen_identity(model, mutable_names):
    """Include every graph field and initializer except explicitly allowed code bytes."""
    value = load_model(model)
    for tensor in value.graph.initializer:
        if tensor.name in mutable_names:
            array = numpy_helper.to_array(tensor)
            if array.dtype != np.int8:
                raise ValueError("Only INT8 weight codes may be mutable")
            tensor.CopyFrom(numpy_helper.from_array(np.zeros_like(array), tensor.name))
    return model_identity(value)


def _array_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _broadcast(array, weight, axis):
    array = np.asarray(array)
    if array.ndim == 0:
        return array
    if axis is None or array.ndim != 1 or len(array) != weight.shape[axis]:
        raise ValueError("Invalid per-channel encoding shape/axis")
    shape = [1] * weight.ndim
    shape[axis] = len(array)
    return array.reshape(shape)


def _initializer_values(model):
    return {v.name: numpy_helper.to_array(v) for v in model.graph.initializer}


def _constant(name, values, producers):
    if name in values:
        return np.array(values[name], copy=True)
    dq = producers.get(name)
    if (
        dq is None
        or dq.op_type != "DequantizeLinear"
        or any(v not in values for v in dq.input)
    ):
        raise ValueError("Expected a constant QDQ weight/bias: " + name)
    codes, scale, zero = [values[v] for v in dq.input]
    axis = next((a.i for a in dq.attribute if a.name == "axis"), None)
    return (
        codes.astype(np.float32) - _broadcast(zero.astype(np.float32), codes, axis)
    ) * _broadcast(scale, codes, axis)


def _targets(original, quantized, audit):
    values = _initializer_values(quantized)
    source_values = _initializer_values(original)
    producers = {name: n for n in quantized.graph.node for name in n.output}
    consumers = {}
    for node in quantized.graph.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node)
    rows = []
    for row in audit["weights"]:
        index = row["node"]
        source = original.graph.node[index]
        attributes(source)
        dq_name = "__matrix_w" + str(index) + "_DequantizeLinear"
        found = [n for n in quantized.graph.node if n.name == dq_name]
        if len(found) != 1:
            raise ValueError("Expected one encoded weight DQ: " + dq_name)
        dq = found[0]
        users = consumers.get(dq.output[0], [])
        if len(users) != 1 or len(consumers.get(dq.input[0], [])) != 1:
            raise ValueError(
                "Shared encoded weight initializer is unsupported: " + dq.input[0]
            )
        student = users[0]
        if student.op_type != source.op_type or student.input[1] != dq.output[0]:
            raise ValueError("Encoded weight consumer differs from original")
        if [a.SerializeToString() for a in source.attribute] != [
            a.SerializeToString() for a in student.attribute
        ]:
            raise ValueError("Weighted operator attributes changed")
        weight = np.array(source_values[source.input[1]], copy=True)
        scale, zero, codes = (
            np.array(values[v], copy=True)
            for v in (dq.input[1], dq.input[2], dq.input[0])
        )
        axis = next((a.i for a in dq.attribute if a.name == "axis"), None)
        if (
            codes.dtype != np.int8
            or weight.dtype != np.float32
            or not np.isfinite(weight).all()
        ):
            raise ValueError("AdaRound requires finite FP32 weights and INT8 codes")
        if (
            codes.shape != weight.shape
            or np.any(zero != 0)
            or np.any(scale <= 0)
            or not np.isfinite(scale).all()
        ):
            raise ValueError("Invalid frozen signed symmetric encoding")
        broadcast = _broadcast(scale, weight, axis)
        with np.errstate(over="ignore", invalid="ignore"):
            quotient = weight / broadcast
        if not np.isfinite(quotient).all():
            raise ValueError("Unrepresentable weight/scale quotient")
        rtn = np.clip(np.rint(quotient), -128, 127).astype(np.int8)
        if not np.array_equal(rtn, codes):
            raise ValueError(
                "Input must be the original RTN codes, not a previously optimized model"
            )
        bias = (
            _constant(student.input[2], values, producers)
            if len(student.input) > 2
            else None
        )
        source_bias = source_values[source.input[2]] if len(source.input) > 2 else None
        rows.append(
            dict(
                index=index,
                source=source,
                student=student,
                codes_name=dq.input[0],
                weight=weight,
                scale=broadcast,
                axis=axis,
                rtn=codes,
                bias=bias,
                source_bias=source_bias,
            )
        )
    return rows


def _prefix(model, outputs):
    """Prune the graph at requested tensors, retaining original operation semantics."""
    inferred = onnx.shape_inference.infer_shapes(
        model, strict_mode=True, data_prop=True
    )
    producers = {
        name: i for i, node in enumerate(inferred.graph.node) for name in node.output
    }
    needed, indices = set(outputs), set()
    pending = list(outputs)
    while pending:
        name = pending.pop()
        i = producers.get(name)
        if i is not None and i not in indices:
            indices.add(i)
            for value in inferred.graph.node[i].input:
                if value and value not in needed:
                    needed.add(value)
                    pending.append(value)
    info = {
        v.name: v
        for v in [
            *inferred.graph.input,
            *inferred.graph.output,
            *inferred.graph.value_info,
        ]
    }
    for tensor in inferred.graph.initializer:
        info.setdefault(
            tensor.name,
            helper.make_tensor_value_info(
                tensor.name, tensor.data_type, list(tensor.dims)
            ),
        )
    if any(v not in info for v in outputs):
        raise ValueError("Shape/type inference missing a reconstruction tensor")
    graph = helper.make_graph(
        [n for i, n in enumerate(inferred.graph.node) if i in indices],
        "adaround_prefix",
        [v for v in inferred.graph.input if v.name in needed],
        [info[v] for v in dict.fromkeys(outputs)],
        [v for v in inferred.graph.initializer if v.name in needed],
    )
    inferred.graph.CopyFrom(graph)
    onnx.checker.check_model(inferred)
    return inferred


def _run(session, names, feed):
    from opennpu_quant.ort.fetch import run_tensors

    return run_tensors(session, names, feed)


def _scan_feeds(factory):
    if not callable(factory):
        raise TypeError(
            "apply_adaround() needs a replayable feed factory: a zero-argument callable "
            "returning a new iterator of {input name: array} dicts on every call "
            "(AdaRound rereads inputs for each layer). Example: "
            "apply_adaround(model, quantized, lambda: iter(feed_list))"
        )
    signatures = []
    for feed in factory():
        if not feed or any(
            np.asarray(a).dtype.hasobject or not np.isfinite(a).all()
            for a in feed.values()
        ):
            raise ValueError("Empty/nonfinite reconstruction feed")
        signatures.append(feed_digest(feed))
    if not signatures:
        raise ValueError("No reconstruction samples")
    return signatures, hashlib.sha256("".join(signatures).encode()).hexdigest()


class _Replay:
    def __init__(self, factory, signatures):
        self.factory, self.signatures = factory, signatures
        self.iterator, self.index = iter(factory()), 0

    def next(self):
        if self.index == len(self.signatures):
            if next(self.iterator, None) is not None:
                raise ValueError("Reconstruction replay added samples")
            self.iterator, self.index = iter(self.factory()), 0
        try:
            value = next(self.iterator)
        except StopIteration:
            raise ValueError("Reconstruction replay lost samples") from None
        if feed_digest(value) != self.signatures[self.index]:
            raise ValueError("Reconstruction replay changed ordered samples")
        self.index += 1
        return value


@contextmanager
def _torch_policy(torch, config, device):
    old = (
        torch.are_deterministic_algorithms_enabled(),
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic,
    )
    workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if device == "cuda" and workspace not in (None, ":4096:8", ":16:8"):
        raise ValueError("Incompatible CUBLAS_WORKSPACE_CONFIG")
    if device == "cuda":
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    devices = [torch.cuda.current_device()] if device == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            yield
        finally:
            torch.use_deterministic_algorithms(old[0])
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = (
                old[1:3]
            )
            torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = old[3:]
            if workspace is None:
                os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
            else:
                os.environ["CUBLAS_WORKSPACE_CONFIG"] = workspace


def rounding_state(weight, scale, *, device):
    """Only alpha requires gradients; exact integer/saturated-equal choices are fixed."""
    import torch

    w = torch.tensor(weight, dtype=torch.float32, device=device)
    s = torch.tensor(scale, dtype=torch.float32, device=device)
    quotient = w / s
    floor = torch.floor(quotient)
    fraction = quotient - floor
    active = (fraction != 0) & (
        torch.clamp(floor, -128, 127) != torch.clamp(floor + 1, -128, 127)
    )
    probability = (fraction + 0.1) / 1.2
    alpha = torch.nn.Parameter(torch.log(probability / (1 - probability)))
    return alpha, floor, s, active


def rounding_codes(alpha, floor, active, *, hard):
    import torch

    h = (
        (alpha >= 0).to(torch.float32)
        if hard
        else torch.clamp(1.2 * torch.sigmoid(alpha) - 0.1, 0, 1)
    )
    h = torch.where(active, h, torch.zeros_like(h))
    return torch.clamp(floor + h, -128, 127)


def _regularizer(alpha, active, step, config):
    import torch

    start = config.steps * config.warmup_fraction
    if step < start:
        return alpha.sum() * 0
    progress = min(1.0, (step - start) / max(1.0, config.steps - 1 - start))
    beta = config.beta_end + 0.5 * (config.beta_start - config.beta_end) * (
        1 + math.cos(math.pi * progress)
    )
    h = torch.clamp(1.2 * torch.sigmoid(alpha) - 0.1, 0, 1)
    return config.regularization * (1 - torch.abs(2 * h[active] - 1).pow(beta)).sum()


# The loss is separable in alpha. Independent leaf blocks avoid full-size
# SliceBackward zero buffers and retain the same per-element gradient formula.
REGULARIZER_BLOCK_ELEMENTS = 262144
REGULARIZER_CHUNK_THRESHOLD = 8388608


def _backward_regularizer(
    alpha,
    active,
    step,
    config,
    *,
    chunk_elements=REGULARIZER_BLOCK_ELEMENTS,
    threshold=REGULARIZER_CHUNK_THRESHOLD,
):
    import torch

    if alpha.numel() <= threshold:
        _regularizer(alpha, active, step, config).backward()
        return
    if step < config.steps * config.warmup_fraction:
        return
    if not alpha.is_contiguous() or not active.is_contiguous():
        raise ValueError(
            "Chunked AdaRound regularization requires contiguous parameters"
        )
    if alpha.grad is None:
        alpha.grad = torch.zeros_like(alpha)
    flat, mask, gradient = alpha.detach().view(-1), active.view(-1), alpha.grad.view(-1)
    for start in range(0, flat.numel(), chunk_elements):
        end = min(flat.numel(), start + chunk_elements)
        block = flat[start:end].detach().clone().requires_grad_(True)
        value = _regularizer(block, mask[start:end], step, config)
        (derivative,) = torch.autograd.grad(value, block)
        with torch.no_grad():
            gradient[start:end].add_(derivative)


def _assert_parity(actual, expected, config, label):
    if (
        actual.shape != expected.shape
        or not np.isfinite(actual).all()
        or not np.allclose(actual, expected, atol=config.atol, rtol=config.rtol)
    ):
        maximum = (
            float(np.max(np.abs(actual - expected)))
            if actual.shape == expected.shape
            else None
        )
        raise ValueError(
            f"Torch/ORT {label} parity failed (max_abs={maximum}, atol={config.atol}, rtol={config.rtol})"
        )


def _fit_layer(
    row,
    original,
    student_model,
    factory,
    signatures,
    config,
    ort,
    device,
    progress=None,
    *,
    cache_config=None,
    profiling=None,
):
    import torch
    from .adaround_cache import LayerActivationCache

    cache = LayerActivationCache(cache_config) if cache_config is not None else None
    clocks = dict(window_seconds=0.0, error_seconds=0.0, training_seconds=0.0)

    def tensor(value):
        return (
            value
            if isinstance(value, torch.Tensor)
            else torch.tensor(value, device=device)
        )

    seed = config.seed + row["index"]
    torch.random.default_generator.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed(seed)
    rng = np.random.default_rng(config.seed + row["index"])
    source, student = row["source"], row["student"]
    teacher_names = [source.input[0], source.output[0]]
    student_names = [student.input[0], student.output[0]]
    teacher = create_session(_prefix(original, teacher_names), ort)
    student_session = create_session(_prefix(student_model, student_names), ort)
    alpha, floor, scale, active = rounding_state(
        row["weight"], row["scale"], device=device
    )
    bias = None if row["bias"] is None else torch.tensor(row["bias"], device=device)
    teacher_bias = (
        None
        if row["source_bias"] is None
        else torch.tensor(row["source_bias"], device=device)
    )
    replay = _Replay(factory, signatures)
    parity_count = 0
    peak_bytes = 0
    indices_seen = set()

    def window(replay):
        nonlocal parity_count, peak_bytes
        started = time.perf_counter()
        items, used = [], 0
        # These constants are needed only for the window's parity checks.
        original_weight = torch.tensor(row["weight"], device=device)
        rtn_weight = torch.tensor(row["rtn"].astype(np.float32), device=device) * scale
        # No pending activation buffers survive a window. Stop before the configured
        # cap using the largest previous pair (static model inputs in this contract).
        max_pair = 0
        for _ in range(min(config.window_samples, len(signatures))):
            if items and used + max_pair > config.window_bytes:
                break
            feed = replay.next()
            key = replay.index - 1
            cached = cache.get(key) if cache is not None else None
            if cached is not None:
                x, y = cached
                pair = x.nbytes + y.nbytes
                max_pair = max(max_pair, pair)
                if pair > config.window_bytes or used + pair > config.window_bytes:
                    raise MemoryError("Activation window exceeds its byte budget")
                indices_seen.add(key)
                items.append(cached)
                used += pair
                continue
            before = feed_digest(feed)
            tx, y = _run(teacher, teacher_names, feed)
            x, sy = _run(student_session, student_names, feed)
            if before != feed_digest(feed):
                raise ValueError("ORT modified reconstruction inputs")
            if any(
                v.dtype != np.float32 or not np.isfinite(v).all()
                for v in (x, y, tx, sy)
            ):
                raise ValueError("Reconstruction requires finite FP32 tensors")
            input_batch_axis = (
                1
                if student.op_type == "Gemm" and attributes(student).get("transA", 0)
                else 0
            )
            if x.shape[input_batch_axis] != 1 or y.shape[0] != 1:
                raise ValueError(
                    "Reconstruction feeds must have batch one for microbatch accumulation"
                )
            pair = x.nbytes + y.nbytes
            max_pair = max(max_pair, pair)
            if pair > config.window_bytes or used + pair > config.window_bytes:
                raise MemoryError("Activation window exceeds its byte budget")
            with torch.no_grad():
                predicted = (
                    operation(student, torch.tensor(x, device=device), rtn_weight, bias)
                    .cpu()
                    .numpy()
                )
                reference = (
                    operation(
                        source,
                        torch.tensor(tx, device=device),
                        original_weight,
                        teacher_bias,
                    )
                    .cpu()
                    .numpy()
                )
            _assert_parity(predicted, sy, config, "RTN " + source.name)
            _assert_parity(reference, y, config, "FP32 " + source.name)
            parity_count += 1
            indices_seen.add(replay.index - 1)
            pair_arrays = (np.array(x, copy=True), np.array(y, copy=True))
            items.append(pair_arrays)
            if cache is not None:
                cache.put(key, pair_arrays)
            used += pair
        peak_bytes = max(peak_bytes, used)
        if cache is not None:
            items = cache.device_window(items, device)
        clocks["window_seconds"] += time.perf_counter() - started
        return items

    def error(items, codes):
        started = time.perf_counter()
        total = 0.0
        with torch.no_grad():
            for x, target in items:
                output = (
                    operation(student, tensor(x), codes * scale, bias).cpu().numpy()
                )
                target_array = (
                    target.cpu().numpy() if isinstance(target, torch.Tensor) else target
                )
                difference = output.astype(np.float64) - target_array.astype(np.float64)
                total += float(
                    np.square(difference).sum(axis=channel_axis(student)).mean()
                )
        clocks["error_seconds"] += time.perf_counter() - started
        return total / len(items)

    items = window(replay)
    initial_error = error(
        items, torch.tensor(row["rtn"].astype(np.float32), device=device)
    )
    optimizer = torch.optim.Adam([alpha], lr=config.learning_rate)
    for step in range(config.steps):
        if step and step % config.window_steps == 0:
            items.clear()
            items = window(replay)
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        indices = rng.integers(0, len(items), size=config.batch_size)
        for i in indices:
            x, y = items[i]
            codes = rounding_codes(alpha, floor, active, hard=False)
            output = operation(student, tensor(x), codes * scale, bias)
            loss = (output - tensor(y)).square().sum(dim=channel_axis(student)).mean()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite AdaRound reconstruction loss")
            (loss / config.batch_size).backward()
        _backward_regularizer(alpha, active, step, config)
        if alpha.grad is None or not torch.isfinite(alpha.grad).all():
            raise ValueError("Nonfinite/missing AdaRound alpha gradient")
        if any(v.grad is not None for v in (floor, scale)):
            raise AssertionError("Only alpha may receive gradients")
        optimizer.step()
        if not torch.isfinite(alpha).all():
            raise ValueError("Nonfinite AdaRound alpha")
        clocks["training_seconds"] += time.perf_counter() - started
        if progress and (step + 1) % config.window_steps == 0:
            progress(step + 1)
    with torch.no_grad():
        codes = rounding_codes(alpha, floor, active, hard=True)
    items.clear()
    items = window(_Replay(factory, signatures))
    final_error = error(items, codes)
    array = codes.cpu().numpy().astype(np.int8)
    validate_hard_codes(row["weight"], row["scale"], array)
    details = dict(
        node=row["index"],
        name=source.name,
        operator=source.op_type,
        source_weight=source.input[1],
        encoded_weight=row["codes_name"],
        axis=row["axis"],
        steps=config.steps,
        initial_code_sha256=_array_hash(row["rtn"]),
        final_code_sha256=_array_hash(array),
        changed_codes=int(np.count_nonzero(array != row["rtn"])),
        trainable_rounding_variables=int(active.sum()),
        initial_hard_reconstruction_error=initial_error,
        final_hard_reconstruction_error=final_error,
        error_scope="fixed first reconstruction window; channel-sum squared error, mean other dimensions",
        error_samples=len(items),
        reconstruction_samples_seen=len(indices_seen),
        parity_samples=parity_count,
        peak_activation_window_bytes=peak_bytes,
        gradient_parameters=["alpha"],
        regularizer_chunked=alpha.numel() > REGULARIZER_CHUNK_THRESHOLD,
        regularizer_block_elements=REGULARIZER_BLOCK_ELEMENTS,
        parity_weights_lifetime="current window only",
    )
    if cache is not None:
        details["activation_cache"] = dict(cache.metrics)
    if profiling is not None:
        profiling.update(clocks, cache=dict(cache.metrics) if cache else None)
    teacher = student_session = None
    del optimizer
    return array, details


def validate_hard_codes(weight, scale, codes):
    """Also validate resumed codes: integer points cannot take the upper neighbor."""
    quotient = weight / scale
    floor = np.floor(quotient)
    low = np.clip(floor, -128, 127)
    high = np.where(quotient == floor, low, np.clip(floor + 1, -128, 127))
    if (
        codes.dtype != np.int8
        or codes.shape != weight.shape
        or not np.all((codes == low) | (codes == high))
    ):
        raise ValueError(
            "Hard codes violate frozen floor/ceil or exact-integer constraints"
        )


def _replace_codes(model, name, array):
    tensor = next(v for v in model.graph.initializer if v.name == name)
    if array.dtype != np.int8 or tuple(tensor.dims) != array.shape:
        raise ValueError("Changed code dtype/shape")
    tensor.CopyFrom(numpy_helper.from_array(array, name))


def apply_adaround(
    fp32_model,
    quantized_result,
    feeds,
    *,
    config: AdaroundConfig,
    ort,
    checkpoint_dir=None,
    activation_cache=None,
):
    """Public, single-writer fixed-encoding weight rounding API."""
    from opennpu_quant._locking import RunLock

    if quantized_result.config.weight_granularity == "per_group":
        raise ValueError("Per-group AdaRound is not supported; use PC AdaRound")
    import torch

    cuda = config.device == "cuda" or (
        config.device == "auto" and "CUDAExecutionProvider" in ort.providers
    )
    index = next(
        (
            int(dict(ort.provider_options[i]).get("device_id", 0))
            if ort.provider_options
            else 0
            for i, provider in enumerate(ort.providers)
            if provider == "CUDAExecutionProvider"
        ),
        0,
    )
    with (
        RunLock(Path(checkpoint_dir) / ".lock")
        if checkpoint_dir is not None
        else nullcontext()
    ):
        with torch.cuda.device(index) if cuda else nullcontext():
            return _apply_adaround(
                fp32_model,
                quantized_result,
                feeds,
                config=config,
                ort=ort,
                checkpoint_dir=checkpoint_dir,
                activation_cache=activation_cache,
            )


def _apply_adaround(
    fp32_model,
    quantized_result,
    feeds,
    *,
    config: AdaroundConfig,
    ort,
    checkpoint_dir=None,
    activation_cache=None,
):
    """Optimize only weight codes; optional explicit layer checkpoints support resume.

    No persistent output is created unless checkpoint_dir is explicitly supplied.
    An incomplete layer restarts from its deterministic layer seed.
    """
    import torch
    from .api import QuantizationResult

    if not isinstance(config, AdaroundConfig):
        raise TypeError("config must be AdaroundConfig")
    from .adaround_cache import ActivationCacheConfig

    if activation_cache is not None and not isinstance(
        activation_cache, ActivationCacheConfig
    ):
        raise TypeError("activation_cache must be ActivationCacheConfig or None")
    if ort.optimization != "ORT_DISABLE_ALL":
        raise ValueError("AdaRound requires ORT_DISABLE_ALL")
    original, model = load_model(fp32_model), load_model(quantized_result.model)
    if quantized_result.audit.get("source_model_identity") != model_identity(original):
        raise ValueError(
            "RTN source identity missing or mismatched; regenerate with quantize()"
        )
    verify_parameters(model, quantized_result.audit)
    rows = _targets(original, model, quantized_result.audit)
    if not rows:
        raise ValueError("No constant weight-bearing operators for AdaRound")
    signatures, samples_identity = _scan_feeds(feeds)
    device = config.device
    if device == "auto":
        device = "cuda" if "CUDAExecutionProvider" in ort.providers else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA AdaRound requested but CUDA Torch is unavailable")
    mutable = {row["codes_name"] for row in rows}
    frozen = frozen_identity(model, mutable)
    execution = environment(ort)
    torch_env = dict(
        torch=torch.__version__,
        cuda=torch.version.cuda,
        device=device,
        device_name=torch.cuda.get_device_name() if device == "cuda" else "CPU",
        cuda_device=execution.get("cuda_device"),
        deterministic=True,
        amp=False,
        tf32=False,
    )
    identity = object_hash(
        [
            model_identity(original),
            model_identity(model),
            config.to_dict(),
            samples_identity,
            execution,
            {
                k: v
                for k, v in torch_env.items()
                if k not in ("device_name", "cuda_device")
            },
            sha256(Path(__file__)),
            sha256(Path(__file__).with_name("adaround_ops.py")),
            sha256(Path(__file__).parents[1] / "ort/fetch.py"),
        ]
    )
    if activation_cache is not None:
        identity = object_hash(
            [
                identity,
                asdict(activation_cache),
                sha256(Path(__file__).with_name("adaround_cache.py")),
            ]
        )
    checkpoint = Path(checkpoint_dir) if checkpoint_dir is not None else None
    completed = []
    saved_codes = {}
    if checkpoint is not None and (checkpoint / "completed.json").exists():
        metadata = read_json(checkpoint / "completed.json")
        if metadata["identity"] != identity:
            raise ValueError(
                "Checkpoint model, feeds, policy or implementation changed; use a new checkpoint directory"
            )
        if metadata["frozen_encoding_identity"] != frozen:
            raise ValueError("Checkpoint frozen encoding differs")
        codes_path = checkpoint / metadata["codes_file"]
        if not codes_path.resolve().is_relative_to(checkpoint.resolve()):
            raise ValueError("Unsafe AdaRound checkpoint path")
        if sha256(codes_path) != metadata["codes_sha256"]:
            raise ValueError("AdaRound layer checkpoint was modified")
        with np.load(codes_path, allow_pickle=False) as archive:
            saved_codes = {name: archive[name].copy() for name in archive.files}
        completed = metadata["layers"]
        expected = [r["codes_name"] for r in rows[: len(completed)]]
        if (
            set(saved_codes) != set(expected)
            or [r["encoded_weight"] for r in completed] != expected
        ):
            raise ValueError("AdaRound checkpoint is not a completed layer prefix")
        for row in completed:
            codes = saved_codes[row["encoded_weight"]]
            if _array_hash(codes) != row["final_code_sha256"]:
                raise ValueError("AdaRound checkpoint code identity mismatch")
            target = next(v for v in rows if v["codes_name"] == row["encoded_weight"])
            validate_hard_codes(target["weight"], target["scale"], codes)
            _replace_codes(model, row["encoded_weight"], codes)

    def status(row, step, state):
        if checkpoint is not None:
            atomic_json(
                checkpoint / "progress.json",
                dict(
                    state=state,
                    identity=identity,
                    updated_at=now(),
                    completed_layers=len(completed),
                    total_layers=len(rows),
                    node=row["index"],
                    step=step,
                    steps=config.steps,
                ),
            )

    with _torch_policy(torch, config, device):
        for row in rows[len(completed) :]:
            status(row, 0, "running")
            try:
                performance = {}
                codes, detail = _fit_layer(
                    row,
                    original,
                    model,
                    feeds,
                    signatures,
                    config,
                    ort,
                    device,
                    progress=lambda step: status(row, step, "running"),
                    cache_config=activation_cache,
                    profiling=performance,
                )
                if activation_cache is not None:
                    detail["cache_timings_seconds"] = performance
                _replace_codes(model, row["codes_name"], codes)
                if frozen_identity(model, mutable) != frozen:
                    raise AssertionError(
                        "AdaRound changed a frozen encoding, bias, axis or graph field"
                    )
                completed.append(detail)
                if checkpoint is not None:
                    saved_codes[row["codes_name"]] = codes
                    checkpoint.mkdir(parents=True, exist_ok=True)
                    temporary = checkpoint / "codes.pending.npz"
                    np.savez_compressed(temporary, **saved_codes)
                    previous = (
                        read_json(checkpoint / "completed.json")
                        if (checkpoint / "completed.json").exists()
                        else {}
                    )
                    digest = sha256(temporary)
                    code_file = "codes_" + digest + ".npz"
                    temporary.replace(checkpoint / code_file)
                    atomic_json(
                        checkpoint / "completed.json",
                        dict(
                            identity=identity,
                            layers=completed,
                            codes_file=code_file,
                            codes_sha256=digest,
                            frozen_encoding_identity=frozen,
                        ),
                    )
                    # The old snapshot remains valid until the new manifest commits.
                    old_name = previous.get("codes_file")
                    old_digest = previous.get("codes_sha256")
                    if (
                        old_name
                        and old_name != code_file
                        and old_name == "codes_" + str(old_digest) + ".npz"
                    ):
                        obsolete = checkpoint / old_name
                        if obsolete.is_file() and sha256(obsolete) == old_digest:
                            obsolete.unlink()
                status(row, config.steps, "layer_completed")
            except BaseException:
                status(row, 0, "interrupted_or_failed")
                raise
    audit = copy.deepcopy(quantized_result.audit)
    audit["adaround"] = dict(
        config=config.to_dict(),
        identity=identity,
        samples_identity=samples_identity,
        sample_count=len(signatures),
        frozen_encoding_identity=frozen,
        layers=completed,
        environment=torch_env,
        initial_parameter_identity=audit["parameter_identity"],
        final_parameter_identity=parameter_identity(model),
        changed_codes=sum(r["changed_codes"] for r in completed),
        state="completed",
        dynamic_matmul=[
            dict(node=i, name=n.name, reason="no constant weight")
            for i, n in enumerate(original.graph.node)
            if n.op_type == "MatMul" and n.input[1] not in _initializer_values(original)
        ],
        scale_selection_audit="unchanged RTN scale-selection evidence; weight MSE candidates precede AdaRound",
    )
    if activation_cache is not None:
        audit["adaround"]["activation_cache_config"] = asdict(activation_cache)
    audit["parameter_identity"] = parameter_identity(model)
    qconfig = replace(quantized_result.config, adaround=config)
    audit["scheme"] = qconfig.scheme()
    verify_parameters(model, audit)
    from .engine import infer_model, layout, verify

    inferred = infer_model(original)
    plan = layout(inferred, qconfig.scope)
    plan["raw_aliases"] = audit.get("raw_aliases", {})
    verify(inferred, model, plan, qconfig.scheme())
    onnx.checker.check_model(model, full_check=True)
    return QuantizationResult(model, qconfig, quantized_result.range_identity, audit)
