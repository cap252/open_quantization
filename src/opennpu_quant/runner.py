from pathlib import Path
import hashlib, tempfile, time, shlex
from uuid import uuid4
from ._io import atomic_json, read_json, object_hash, sha256, now
from ._locking import RunLock
from .configuration import validate, runtime, scheme_for
from .models.spec import ModelSpec
from .models.bundle import ModelBundle
from .models.adapter import ModelEvaluator
from .data.samples import Dataset, preprocessing_implementation
from .graph.model import model_identity, save_model
from .api import calibrate, quantize, apply_adaround
from .quant.calibration import feed_digest
from .quant.storage import save_statistics, read_statistics
from .quant.config import AdaroundConfig, CalibrationConfig
from .quant.adaround_cache import ActivationCacheConfig
from .ort.session import environment, create_session
from .evaluation.loop import evaluate
from .report import report


_REUSE_GUIDANCE = (
    "Use a new --output to keep previous results, or --force to recompute selected "
    "results (calibration caches and AdaRound checkpoints may still be reused)."
)


def _config_changes(before, after, prefix=""):
    if isinstance(before, dict) and isinstance(after, dict):
        paths = []
        for key in sorted(before.keys() | after.keys(), key=str):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in before or key not in after:
                paths.append(path)
            else:
                paths.extend(_config_changes(before[key], after[key], path))
        return sorted(paths)
    return [prefix or "<root>"] if before != after else []


def _code_identity():
    base = Path(__file__).parent
    return object_hash(
        {str(p.relative_to(base)): sha256(p) for p in sorted(base.rglob("*.py"))}
    )


def runtime_audit(result, ort):
    from .quant.engine import audit_runtime

    with tempfile.TemporaryDirectory(
        dir=ort.scratch, prefix="opennpu_audit_"
    ) as folder:
        original = Path(folder) / "qdq.onnx"
        executed = Path(folder) / "executed.onnx"
        save_model(result.model, original)
        session = create_session(original, ort, optimized=executed)
        del session
        return audit_runtime(original, executed, result.audit)


def run(config, paths, *, force=False, recalibrate=False):
    """Execute configured models sequentially under one run-directory lock."""
    validate(config, models_dir=paths.models)
    if recalibrate:
        if config["calibration"]["source"] != "compute":
            raise ValueError("Use calibration.source=compute with --recalibrate")
        if not any(config["conditions"].values()):
            raise ValueError("--recalibrate requires a quantized condition")
    out = paths.runs / config["name"]
    ort = runtime(config)
    code = _code_identity()
    with RunLock(out / ".lock"):
        recorded = out / "effective_config.json"
        if recorded.exists() and not force:
            previous = read_json(recorded)
            if previous != config:
                changed = _config_changes(previous, config)
                detail = ", ".join(changed[:10])
                if len(changed) > 10:
                    detail += f" (+{len(changed) - 10} more)"
                raise ValueError(
                    f"Run configuration changed. Changed keys: {detail}. "
                    + _REUSE_GUIDANCE
                )
        atomic_json(out / "effective_config.json", config)
        status = dict(
            state="running",
            attempt_id=uuid4().hex,
            started=now(),
            finished=None,
            models={
                name: dict(
                    state="pending",
                    conditions={c: {"state": "pending"} for c in config["conditions"]},
                )
                for name in config["models"]
            },
        )
        atomic_json(out / "status.json", status)
        try:
            if recalibrate:
                for name in config["models"]:
                    atomic_json(
                        out / name / "recalibration.json",
                        dict(generation=status["attempt_id"], state="pending"),
                    )
            for name in config["models"]:
                status["models"][name]["state"] = "running"
                atomic_json(out / "status.json", status)
                _run_model(name, config, paths, out, ort, code, force, status)
                status["models"][name]["state"] = "completed"
                atomic_json(out / "status.json", status)
                report(out)
        except BaseException as error:
            status["state"] = (
                "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
            )
            status["error"] = dict(type=type(error).__name__, message=str(error))
            status["finished"] = now()
            for state in status["models"].values():
                if state["state"] == "running":
                    state.update(state=status["state"], error=status["error"])
                for condition in state["conditions"].values():
                    if condition["state"] == "running":
                        condition.update(state=status["state"], error=status["error"])
            atomic_json(out / "status.json", status)
            report(out)
            raise
        status.update(state="completed", finished=now())
        atomic_json(out / "status.json", status)
        return report(out)


def _run_model(name, config, paths, out, ort, code, force, status):
    local_recipe = paths.models / name / "recipe.yaml"
    spec = ModelSpec.load(local_recipe if local_recipe.exists() else name)
    bundle = ModelBundle(spec, paths.models / name)
    dataset_name = spec.recipe["dataset"]
    manifest = paths.data / dataset_name / "manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(
            f"Dataset '{dataset_name}' is not prepared (missing {manifest}). "
            f"Run: opennpu-quant data prepare {shlex.quote(dataset_name)} "
            f"--root '<dataset folder>' --data-dir {shlex.quote(str(paths.data))}"
        )
    dataset = Dataset(
        manifest,
        root=paths.datasets.get(dataset_name),
    )
    expected = dataset.info["expected_samples"]
    limit = config["evaluation"]["limit"]
    network = bundle.paths["network"]
    # Hash the materialized graph, including the bytes actually read from external data.
    # Bundle location and external-data filenames are not model identity.
    graphs = {k: model_identity(v) for k, v in bundle.paths.items()}
    model_key = graphs["network"]
    implementation = preprocessing_implementation(spec.recipe["preprocess"])
    preprocessing_identity = (
        object_hash(
            [
                spec.recipe["preprocess"],
                spec.recipe["input"],
                bundle.mapping,
                graphs.get("preprocess"),
                implementation,
            ]
        )
        if implementation
        else None
    )
    evaluation_identity = (
        object_hash(
            [
                dataset.identity("validation", limit),
                spec.recipe,
                environment(ort),
                code,
                {k: v for k, v in graphs.items() if k != "network"},
                bundle.mapping,
                preprocessing_identity,
            ]
        )
        if preprocessing_identity
        else None
    )
    count = config["calibration"]["samples"][spec.recipe["dataset"]]
    factory = dataset.feeds(bundle, count)
    calibration_data_identity = dataset.identity("calibration", count)
    statistics = None
    input_cache_evidence = None
    rtns = {}
    request_path = out / name / "recalibration.json"
    request = (
        read_json(request_path)
        if request_path.exists() and config["calibration"]["source"] == "compute"
        else None
    )
    generation = request["generation"] if request else None
    for condition, policy in config["conditions"].items():
        current = status["models"][name]["conditions"][condition]
        current["state"] = "running"
        atomic_json(out / "status.json", status)
        start = time.monotonic()
        directory = out / name / condition
        scheme = scheme_for(config, name, policy["scheme"]) if policy else None
        identity = object_hash(
            [
                code,
                graphs,
                bundle.mapping,
                evaluation_identity,
                condition,
                policy,
                scheme.to_dict() if scheme else None,
                config["adaround"] if policy.get("adaround") else None,
                config.get("activation_cache"),
                count,
                calibration_data_identity,
                config["calibration"],
                limit,
            ]
        )
        complete = directory / "complete.json"
        target = directory / "result.json"
        if complete.exists() and not force:
            if not preprocessing_identity:
                raise ValueError(
                    "Cannot reuse results with unidentified custom preprocessing. "
                    + _REUSE_GUIDANCE
                )
            proof = read_json(complete)
            if proof["identity"] != identity:
                raise ValueError(
                    f"Run execution identity changed for condition '{condition}'. "
                    "The stored identity does not identify the exact cause. "
                    "Possible causes include code, model/host graphs, inputs or "
                    "preprocessing, runtime or quantization settings. "
                    + _REUSE_GUIDANCE
                )
            if target.is_file() and sha256(target) == proof["result_sha256"]:
                payload = directory / "model.onnx"
                model_ok = (
                    not proof.get("model_sha256")
                    or payload.is_file()
                    and sha256(payload) == proof["model_sha256"]
                )
                extras_ok = set(proof.get("files", {})) <= {
                    "quantization.json",
                    "predictions.jsonl.gz",
                } and all(
                    (directory / name).is_file() and sha256(directory / name) == digest
                    for name, digest in proof.get("files", {}).items()
                )
                generation_ok = (
                    not policy or proof.get("calibration_generation") == generation
                )
                if model_ok and extras_ok and generation_ok:
                    previous = read_json(target)
                    current.update(
                        state="reused",
                        identity=identity,
                        result_sha256=proof["result_sha256"],
                        measurement_attempt_id=previous.get("attempt_id"),
                        measurement_finished_at=previous.get("finished_at"),
                        verification="same_execution_identity_and_artifact_hashes",
                    )
                    atomic_json(out / "status.json", status)
                    continue
            # A tampered or incomplete result is recomputed from verified inputs.
        if policy:
            if statistics is None:
                input_policy = config["calibration"].get("input_cache", {})
                if input_policy.get("enabled", False):
                    cache_start = time.monotonic()
                    factory = dataset.cached_feeds(
                        bundle,
                        count,
                        paths.cache / "inputs" / name,
                        max_bytes=input_policy.get("max_bytes", 2 * 1024**3),
                    )
                    input_cache_evidence = dict(
                        identity=factory.manifest["identity"],
                        samples_identity=factory.manifest["samples_identity"],
                        sample_count=factory.manifest["sample_count"],
                        bytes=sum(v["bytes"] for v in factory.manifest["tensors"]),
                        reused=factory.reused,
                        prepare_seconds=time.monotonic() - cache_start,
                        manifest_sha256=sha256(factory.directory / "manifest.json"),
                    )
                    atomic_json(out / name / "input_cache.json", input_cache_evidence)
                signatures = hashlib.sha256()
                for feed in factory():
                    signatures.update(feed_digest(feed).encode())
                feed_id = signatures.hexdigest()
                calibort = (
                    ort
                    if config["calibration"]["device"] == "run"
                    else runtime(config, config["calibration"]["device"])
                )
                key = object_hash(
                    [
                        model_key,
                        feed_id,
                        config["calibration"]["histogram_bins"],
                        environment(calibort),
                        {
                            p: sha256(Path(__file__).parent / "quant" / p)
                            for p in ("calibration.py", "histograms.py", "engine.py")
                        },
                    ]
                )
                cache = paths.cache / "calibration" / name / key
                statsconfig = CalibrationConfig(
                    histogram_bins=config["calibration"]["histogram_bins"]
                )
                reference = config["calibration"]["source"] == "reference"
                if request:
                    cache = cache / "requests" / generation
                if reference:
                    cache = paths.cache / "calibration" / name / "reference"
                    if not (cache / "complete.json").exists():
                        raise FileNotFoundError(
                            "Verified reference statistics bundle is missing: "
                            + str(cache)
                        )
                pending = request is not None and request["state"] == "pending"
                if (cache / "complete.json").exists() and not pending:
                    statistics = read_statistics(
                        cache,
                        network,
                        config=statsconfig,
                        ort=calibort,
                        samples_identity=feed_id,
                    )
                else:
                    statistics = calibrate(
                        network, factory, config=statsconfig, ort=calibort
                    )
                    save_statistics(statistics, cache)
                if request:
                    if (
                        not pending
                        and request["statistics_identity"]
                        != statistics.statistics_identity
                    ):
                        raise ValueError(
                            "Recalibration statistics changed; request --recalibrate again"
                        )
                    request.update(
                        state="completed",
                        statistics_identity=statistics.statistics_identity,
                    )
                    atomic_json(request_path, request)
            family = policy["scheme"]
            if family not in rtns:
                # Use the collection execution profile to validate encodings, independent of evaluation device.
                rtns[family] = quantize(network, statistics, scheme, ort=calibort)
            result = rtns[family]
            if policy.get("adaround"):
                opts = dict(config["adaround"])
                opts.pop("data", None)
                cc = dict(config.get("activation_cache", {}))
                enabled = cc.pop("enabled", True)
                result = apply_adaround(
                    network,
                    result,
                    factory,
                    AdaroundConfig(**opts),
                    ort=ort,
                    checkpoint_dir=(
                        directory / "adaround" / identity / generation
                        if generation
                        else directory / "adaround" / identity
                    ),
                    activation_cache=ActivationCacheConfig(**cc) if enabled else None,
                )
            audit = runtime_audit(result, ort)
            model = result.model
        else:
            model = network
            result = None
            audit = None
        evaluator = ModelEvaluator(bundle, dataset)
        # Optional prediction stream never retains the full preprocessed dataset.
        stream = None
        if config["output"]["save_predictions"]:
            import gzip, json

            directory.mkdir(parents=True, exist_ok=True)
            stream = gzip.open(directory / "predictions.jsonl.gz.pending", "wt")
            original = evaluator.update

            def update(outputs, sample):
                original(outputs, sample)
                stream.write(
                    json.dumps(
                        dict(id=sample.sample_id, prediction=evaluator.prediction)
                    )
                    + "\n"
                )

            evaluator.update = update
        try:
            values = evaluate(
                model,
                dataset.samples(bundle, limit=limit),
                evaluator=evaluator,
                ort=ort,
                expected_samples=expected,
                limit=limit,
            )
        finally:
            if stream:
                stream.close()
        if stream:
            (directory / "predictions.jsonl.gz.pending").replace(
                directory / "predictions.jsonl.gz"
            )
        metric = spec.recipe["metric"]
        record = dict(
            attempt_id=status["attempt_id"],
            model=name,
            title=spec.recipe["title"],
            model_metadata={
                k: spec.recipe[k]
                for k in (
                    "source",
                    "dataset",
                    "task",
                    "input",
                    "metric",
                    "preprocess",
                    "decode",
                )
            },
            condition=condition,
            metric=metric,
            value=values.metrics[metric],
            metrics=values.metrics,
            metric_units=values.metric_units,
            metric_details=values.metric_details,
            samples=values.samples,
            expected_samples=expected,
            complete=values.complete,
            reference=bool(dataset.info.get("reference_bytes_verified"))
            and values.complete,
            config_identity=object_hash(config),
            evaluation_identity=evaluation_identity,
            preprocessing_identity=preprocessing_identity,
            identity=identity,
            model_identity=model_key,
            graph_identities=graphs,
            scheme=scheme.to_dict() if scheme else None,
            calibration_samples=count if policy else None,
            calibration_identity=statistics.statistics_identity if policy else None,
            input_cache=input_cache_evidence if policy else None,
            adaround=result.audit.get("adaround") if result else None,
            runtime=values.runtime,
            runtime_audit=audit,
            elapsed_s=time.monotonic() - start,
            finished_at=now(),
        )
        if policy and generation:
            record["calibration_generation"] = generation
        payload_hash = None
        if result and config["output"]["save_qdq_models"]:
            save_model(result.model, directory / "model.onnx")
            payload_hash = sha256(directory / "model.onnx")
        if result:
            atomic_json(directory / "quantization.json", result.audit)
        atomic_json(target, record)
        artifacts = {
            name: sha256(directory / name)
            for name in ("quantization.json", "predictions.jsonl.gz")
            if (directory / name).is_file()
        }
        atomic_json(
            complete,
            dict(
                identity=identity,
                result_sha256=sha256(target),
                model_sha256=payload_hash,
                files=artifacts,
                **(
                    {"calibration_generation": generation}
                    if policy and generation
                    else {}
                ),
            ),
        )
        current.update(
            state="completed",
            identity=identity,
            result_sha256=sha256(target),
            measurement_attempt_id=status["attempt_id"],
        )
        atomic_json(out / "status.json", status)
