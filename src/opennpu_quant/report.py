from pathlib import Path
from ._io import read_json, atomic_json, sha256, object_hash
from .results import (
    export_records,
    metric_values,
    normalize_scheme,
    normalize_formats,
    report_policy,
    percent,
    tables,
)


def report(directory, *, formats=None):
    """Compare matched FP32 denominators without treating missing results as zero."""
    selected_formats = normalize_formats(formats, allowed=("md", "csv", "tsv", "json"))
    directory = Path(directory)
    if not (directory / "effective_config.json").is_file():
        raise ValueError(
            f"Not a run directory: {directory} (effective_config.json is missing). "
            "Pass <output>/<experiment name>, e.g. workspace/runs/core10"
        )
    config = read_json(directory / "effective_config.json")
    effective_policy = report_policy(config.get("report"))
    records = []
    means = {}
    status_path = directory / "status.json"
    status = read_json(status_path) if status_path.is_file() else {}

    def verified(path):
        proof_path = path.parent / "complete.json"
        if not path.is_file() or not proof_path.is_file():
            return False
        proof = read_json(proof_path)
        if proof.get("result_sha256") != sha256(path):
            return False
        measured = read_json(path)
        if measured.get("config_identity") != object_hash(config):
            return False
        if status.get("attempt_id"):
            current = (
                status.get("models", {})
                .get(path.parent.parent.name, {})
                .get("conditions", {})
                .get(path.parent.name, {})
            )
            if (
                current.get("state") not in {"completed", "reused"}
                or current.get("result_sha256") != proof.get("result_sha256")
                or current.get("identity") != measured.get("identity")
                or proof.get("identity") != measured.get("identity")
                or current.get("measurement_attempt_id") != measured.get("attempt_id")
            ):
                return False
        if set(proof.get("files", {})) - {"quantization.json", "predictions.jsonl.gz"}:
            return False
        for name, digest in proof.get("files", {}).items():
            payload = path.parent / name
            if not payload.is_file() or sha256(payload) != digest:
                return False
        if proof.get("model_sha256"):
            model = path.parent / "model.onnx"
            if not model.is_file() or sha256(model) != proof["model_sha256"]:
                return False
        return True

    for model in config["models"]:
        for condition in config["conditions"]:
            path = directory / model / condition / "result.json"
            ok = verified(path)
            current = (
                status.get("models", {})
                .get(model, {})
                .get("conditions", {})
                .get(condition, {})
            )
            row = read_json(path) if ok else {}
            metadata = row.get("model_metadata", {})
            binding = "recorded_at_measurement" if metadata else "not_recorded"
            if not metadata:
                # A reference recipe is not measured provenance.
                try:
                    from .models.spec import ModelSpec

                    recipe = ModelSpec.load(model).recipe
                    metadata = {
                        k: recipe[k]
                        for k in ("source", "dataset", "task", "input", "metric")
                    }
                    binding = "packaged_recipe_reference"
                except (FileNotFoundError, ValueError):
                    pass
            policy = config["conditions"][condition]
            scheme = row.get("scheme")
            if not ok and policy:
                scheme = dict(config["schemes"][policy["scheme"]])
                if scheme.get("percentile") == "per_model":
                    scheme["percentile"] = config["percentiles"][model][
                        policy["scheme"]
                    ]
            ada = (row.get("adaround") or {}).get("config")
            if ada is None and policy.get("adaround"):
                ada = config["adaround"]
            task = metadata.get("task")
            if ok and task is None and effective_policy["primary_metric"]:
                raise ValueError(
                    "Report primary_metric requires model task metadata: " + model
                )
            primary = effective_policy["primary_metric"].get(
                task, row.get("metric", metadata.get("metric", "unknown"))
            )
            metrics = metric_values(row.get("metrics", {}), row.get("metric_units"))
            if ok and (primary not in metrics or percent(metrics[primary]) is None):
                raise ValueError(
                    "Missing or incompatible report metric: " + model + "/" + primary
                )
            records.append(
                dict(
                    run_id=config["name"],
                    model=model,
                    model_name=row.get("title", model),
                    condition=condition,
                    state="completed"
                    if ok
                    else (
                        current.get(
                            "state", "unverified" if path.exists() else "pending"
                        )
                        if current.get("state") not in {"completed", "reused"}
                        else "unverified"
                    ),
                    origin="standalone_run",
                    primary_metric=primary,
                    metrics=metrics,
                    metric_details=row.get("metric_details", {}),
                    dataset=metadata.get("dataset"),
                    split=config["evaluation"]["split"],
                    samples=row.get("samples"),
                    expected_samples=row.get("expected_samples"),
                    complete=row.get("complete", False),
                    reference=row.get("reference", False),
                    evaluation_identity=row.get("evaluation_identity"),
                    quantization=normalize_scheme(scheme, ada),
                    model_source=metadata.get("source", {}),
                    model_metadata=metadata,
                    runtime=row.get("runtime", {}),
                    finished_at=row.get("finished_at"),
                    elapsed_s=row.get("elapsed_s"),
                    evidence=dict(
                        result_sha256=sha256(path) if ok else None,
                        result_hash_verified=ok,
                        source_binding=binding,
                        model_identity=row.get("model_identity"),
                        preprocessing_identity=row.get("preprocessing_identity"),
                        identity=row.get("identity"),
                        attempt_id=status.get("attempt_id"),
                        measurement_attempt_id=row.get("attempt_id"),
                        current_attempt=current,
                    ),
                )
            )
    rows = tables(records)[0]
    baselines = {r["model"]: r for r in rows if r["condition"] == "fp32"}
    for condition in config["conditions"]:
        selected = [r for r in rows if r["condition"] == condition]
        means[condition] = (
            sum(r["recovery_pct"] for r in selected) / len(selected)
            if len(selected) == len(config["models"])
            and all(
                r["complete"]
                and r["samples"] == r["expected_samples"]
                and r["recovery_pct"] is not None
                and baselines.get(r["model"], {}).get("complete", False)
                and baselines[r["model"]]["samples"]
                == baselines[r["model"]]["expected_samples"]
                == r["expected_samples"]
                for r in selected
            )
            else None
        )
    target = effective_policy["target_mean_recovery"]
    target_met = {
        c: value >= target if value is not None and target is not None else None
        for c, value in means.items()
    }
    lines = [
        "# Evaluation results",
        "",
        "QDQ / matched FP32 × 100; arithmetic mean with equal weight per model.",
        "",
        "| Model | Condition | Metric | Accuracy (%) | Recovery (%) | Samples | Complete |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for r in rows:
        ratio = "—" if r["recovery_pct"] is None else f"{r['recovery_pct']:.4f}"
        accuracy = "—" if r["value_pct"] is None else f"{r['value_pct']:.4f}"
        lines.append(
            f"| {r['model']} | {r['condition']} | {r['metric']} | {accuracy} | {ratio} | {r['samples']} | {r['complete']} |"
        )
    lines += ["", "## Complete cohort mean", ""] + [
        f"- {c}: {v:.4f}%"
        if v is not None
        else f"- {c}: incomplete / unmatched denominator"
        for c, v in means.items()
    ]
    if target is not None:
        lines += ["", f"Target mean recovery: {target:g}%", ""]
        lines += [
            f"- {c}: "
            + ("not evaluated" if value is None else "met" if value else "not met")
            for c, value in target_met.items()
        ]
    lines += [
        "",
        "These metrics describe QDQ accuracy. Integer-kernel or shift execution is not implied.",
        "A complete local cohort does not certify equivalence to published/reference dataset bytes.",
    ]
    result = dict(
        rows=rows,
        mean_recovery=means,
        report_policy=effective_policy,
        target_met=target_met,
    )
    table_formats = selected_formats - {"md"}
    if table_formats:
        export_records(
            records,
            directory,
            formats=table_formats,
            metadata={
                "kind": "standalone_run",
                "config_identity": object_hash(config),
                "report_policy": effective_policy,
            },
        )
    if "md" in selected_formats:
        (directory / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if "json" in selected_formats:
        atomic_json(directory / "summary.json", result)
    return result
