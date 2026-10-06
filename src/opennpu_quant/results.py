"""Measurement records and spreadsheet exports with explicit metric units."""

import csv
import io
import json
import math
from pathlib import Path
from ._io import atomic_json, read_json, object_hash
from ._identity import compare_identity_evidence

SCHEMA_VERSION = 1
SUCCESS = {"completed", "evaluated", "reused"}
METRIC_FIELDS = "run_id model model_name condition state metric value unit value_pct primary".split()

RAW_FIELDS = """model task dataset metric fp32_accuracy int8_accuracy recovery_percent
scheme percentile model_source_url model_source_revision top5_percent AP50_percent AP75_percent
metrics_json""".split()


def compact(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def metric_values(values, units=None):
    """Known current evaluators return fractions; importers must supply their units."""
    units = units or {}
    result = {}
    for key, value in values.items():
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (float, int))
            or not math.isfinite(value)
        ):
            raise ValueError("Invalid metric: " + key)
        result[key] = {"value": value, "unit": units.get(key, "fraction")}
    return result


def percent(metric):
    if not metric or metric.get("value") is None:
        return None
    value = metric["value"]
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError("Nonfinite or nonnumeric metric")
    if metric.get("unit") == "fraction":
        return 100 * value
    if metric.get("unit") == "percent":
        return value
    return None  # Unknown units and e.g. dB must not silently become percentages.


def normalize_scheme(value, adaround=None):
    """Normalize serialized Scheme or QuantizationConfig settings."""
    if value is None:
        return None
    v = dict(value)
    c = v.get("calibration", {})
    return dict(
        scope=v.get("scope"),
        representation=v.get("representation", "QDQ"),
        activation_dtype=(
            "int" if v.get("activation_precision", {}).get("signed", True) else "uint"
        )
        + str(v.get("activation_precision", {}).get("bits", 8)),
        weight_dtype=(
            "int" if v.get("weight_precision", {}).get("signed", True) else "uint"
        )
        + str(v.get("weight_precision", {}).get("bits", 8)),
        bias_dtype="int32",
        activation_symmetric=v.get("activation_symmetric"),
        weight_symmetric=v.get("weight_symmetric", True),
        activation_granularity=v.get("activation_granularity", "per_tensor"),
        weight_granularity=v.get("weight_granularity"),
        weight_group_size=v.get("weight_group_size"),
        calibration=dict(
            method=c.get("method", v.get("method")),
            percentile=c.get("percentile_value", v.get("percentile")),
            histogram_bins=c.get("histogram_bins", v.get("histogram_bins")),
            quantized_bins=c.get("quantized_bins", v.get("quantized_bins")),
        ),
        activation_scale=v.get(
            "activation_scale_policy", v.get("activation_scale", "float")
        ),
        weight_scale=v.get("weight_scale_policy", v.get("weight_scale", "float")),
        adaround=v.get("adaround") or adaround,
    )


def scheme_label(q):
    if not q:
        return "FP32"
    cal = q["calibration"]
    method = str(cal.get("method"))
    if method.lower() == "percentile":
        method = "P" + str(cal.get("percentile"))
    a = "sym" if q.get("activation_symmetric") else "asym"
    return "/".join(
        map(
            str,
            [
                q.get("scope"),
                "A:" + q["activation_dtype"] + ":" + a,
                "W:" + q["weight_dtype"] + ":" + str(q.get("weight_granularity")),
                method,
                q.get("activation_scale"),
                q.get("weight_scale"),
                "AdaRound" if q.get("adaround") else "RTN",
            ],
        )
    )


def raw_scheme(row):
    """Describe a policy group, leaving percentile values in their own column."""
    q = json.loads(row.get("quantization_json") or "null") or {}
    if not q and row.get("condition") == "fp32":
        return "FP32"
    cal = q.get("calibration", {})
    symmetry = {True: "symmetric", False: "asymmetric"}
    parts = dict(
        scope=row.get("scope"),
        activation=symmetry.get(row.get("activation_symmetric")),
        weight=row.get("weight_granularity"),
        activation_scale=row.get("activation_scale"),
        weight_scale=row.get("weight_scale"),
        calibration=(row.get("calibration") or cal.get("method") or "unknown").lower(),
    )
    parts["rounding"] = {"rtn": "RTN", "adaround": "AdaRound"}.get(
        (row.get("rounding") or "unknown").lower(), "unknown"
    )
    # Keep additional policies distinct without adding table columns.
    for key, default in (
        ("activation_dtype", "int8"),
        ("weight_dtype", "int8"),
        ("bias_dtype", "int32"),
        ("activation_granularity", "per_tensor"),
        ("weight_symmetric", True),
    ):
        if q.get(key) is not None and q[key] != default:
            parts[key] = q[key]
    for key in ("weight_group_size", "entropy_implementation"):
        value = row.get(key, q.get(key, cal.get(key)))
        if value is not None:
            parts[key] = value
    for key in ("bias_correction", "activation_recalibration"):
        if row.get(key):
            parts[key] = True
    return " | ".join(
        f"{key}={'unknown' if value is None else str(value).lower() if isinstance(value, bool) else value}"
        for key, value in parts.items()
    )


def raw_export_row(row):
    """Project run or comparison details onto the shared spreadsheet schema."""
    metrics = json.loads(row.get("metrics_json") or "{}")
    q = json.loads(row.get("quantization_json") or "null") or {}
    method = row.get("calibration") or q.get("calibration", {}).get("method") or ""
    values = dict(row)
    values.update(
        model=row.get("model_name", row["model"]),
        fp32_accuracy=row.get("fp32_accuracy_percent", row.get("fp32_pct")),
        int8_accuracy=row.get("int8_accuracy_percent", row.get("int8_pct")),
        recovery_percent=row.get("recovery_percent", row.get("recovery_pct")),
        scheme=raw_scheme(row),
        percentile=row.get("percentile") if method.lower() == "percentile" else None,
        top5_percent=percent(metrics.get("top5")),
        AP50_percent=percent(metrics.get("AP50")),
        AP75_percent=percent(metrics.get("AP75")),
        metrics_json=compact(metrics),
    )
    if values["scheme"] == "FP32":
        values["fp32_accuracy"] = (
            percent(metrics.get(row["metric"])) if row.get("state") in SUCCESS else None
        )
    if row.get("state") not in SUCCESS:
        values.update(
            int8_accuracy=None,
            recovery_percent=None,
            top5_percent=None,
            AP50_percent=None,
            AP75_percent=None,
            metrics_json="{}",
        )
    return {field: values.get(field) for field in RAW_FIELDS}


def tables(records):
    lookup = {}
    for r in records:
        key = (r["run_id"], r["model"], r["condition"])
        if key in lookup:
            raise ValueError("Duplicate measurement record: " + str(key))
        lookup[key] = r
        metric_values({k: m["value"] for k, m in r.get("metrics", {}).items()})
    rows, details = [], []
    for r in records:
        ok = r["state"] in SUCCESS
        metric = r["primary_metric"]
        measured = r.get("metrics", {}) if ok else {}
        value = percent(measured.get(metric))
        b = lookup.get((r["run_id"], r["model"], "fp32"))

        def pairing_identity(row):
            evidence = row.get("evidence", {})
            identity = dict(
                source_model_identity=evidence.get("model_identity"),
                preprocessing_identity=evidence.get("preprocessing_identity"),
                evaluation_identity=row.get("evaluation_identity"),
                metric=row.get("primary_metric"),
                samples=row.get("samples"),
                dataset=row.get("dataset"),
                split=row.get("split"),
            )
            if row.get("origin") == "historical_research":
                # Older evaluation_identity did not cover the execution protocol.
                identity.update(
                    code_identity=evidence.get("source_code_identity"),
                    environment_identity=object_hash(evidence["runtime"])
                    if evidence.get("runtime")
                    else None,
                )
            return identity

        pair_state, pair_reason = compare_identity_evidence(
            pairing_identity(r), pairing_identity(b or {})
        )
        if b and b.get("quantization") is not None:
            pair_state, pair_reason = "mismatch", "mismatch:baseline_is_not_FP32"
        matched = bool(ok and b and b["state"] in SUCCESS and pair_state == "passed")
        if not matched and pair_state == "passed":
            pair_state, pair_reason = (
                "insufficient_evidence",
                "insufficient_evidence:completed_measurement",
            )
        pending_base = bool(
            not ok
            and b
            and b.get("quantization") is None
            and b["state"] in SUCCESS
            and b["primary_metric"] == metric
            and b.get("dataset") == r.get("dataset")
            and b.get("split") == r.get("split")
        )
        base_metrics = b.get("metrics", {}) if matched or pending_base else {}
        fp = percent(base_metrics.get(metric))
        q = r.get("quantization") or {}
        source = r.get("model_source") or {}
        row = dict(
            model=r["model"],
            condition=r["condition"],
            metric=metric,
            value_pct=value,
            fp32_pct=fp,
            delta_pp=value - fp if value is not None and fp is not None else None,
            recovery_pct=100 * value / fp
            if value is not None and fp is not None and fp > 0
            else None,
            samples=r.get("samples"),
            expected_samples=r.get("expected_samples"),
            complete=bool(ok and r.get("complete")),
            reference=bool(r.get("reference", False)),
            percentile=q.get("calibration", {}).get("percentile"),
            activation_scale=q.get("activation_scale"),
            weight_scale=q.get("weight_scale"),
            run_id=r["run_id"],
            model_name=r.get("model_name", r["model"]),
            state=r["state"],
            origin=r["origin"],
            dataset=r.get("dataset"),
            task=r.get("task") or r.get("model_metadata", {}).get("task"),
            split=r.get("split"),
            int8_pct=value if q else None,
            metric_unit=measured.get(metric, {}).get("unit"),
            scheme=scheme_label(r.get("quantization")),
            scope=q.get("scope"),
            activation_dtype=q.get("activation_dtype"),
            activation_symmetric=q.get("activation_symmetric"),
            weight_dtype=q.get("weight_dtype"),
            weight_symmetric=q.get("weight_symmetric"),
            weight_granularity=q.get("weight_granularity"),
            rounding=("AdaRound" if q.get("adaround") else "RTN") if q else None,
            model_source_url=source.get("repo") or source.get("url"),
            model_source_revision=source.get("commit") or source.get("framework"),
            checkpoint_url=source.get("url"),
            checkpoint_sha256=source.get("sha256"),
            metrics_json=compact(measured),
            fp32_metrics_json=compact(base_metrics),
            metric_details_json=compact(r.get("metric_details", {})),
            quantization_json=compact(r.get("quantization")),
            model_source_json=compact(source),
            evidence_json=compact(r.get("evidence", {})),
            pairing_status=pair_state,
            pairing_reason=pair_reason,
        )
        if "comparison" in r:
            comparison = r["comparison"]
            row.update(
                condition_family=comparison["family"],
                activation_recalibration=comparison["activation_recalibration"],
                evaluation_batch_size=comparison["evaluation_batch_size"],
                optimization=comparison["optimization"],
                selection_basis=comparison["selection_basis"],
            )
            if comparison["activation_recalibration"]:
                row["scheme"] += "/activation_recalibration"
        rows.append(row)
        for key, item in measured.items():
            details.append(
                dict(
                    run_id=r["run_id"],
                    model=r["model"],
                    model_name=row["model_name"],
                    condition=r["condition"],
                    state=r["state"],
                    metric=key,
                    value=item["value"],
                    unit=item["unit"],
                    value_pct=percent(item),
                    primary=key == metric,
                )
            )
    return rows, details


def _cell(value):
    if not isinstance(value, str):
        return value
    # Keep one physical row per record and prevent spreadsheet formula execution.
    value = value.replace("\r", "\\r").replace("\n", "\\n").replace("\t", "\\t")
    if value.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def table_text(rows, fields=RAW_FIELDS, *, delimiter="\t"):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream, fieldnames=fields, delimiter=delimiter, lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows({k: _cell(r.get(k)) for k in fields} for r in rows)
    return stream.getvalue()


def _text(path, text, encoding):
    temporary = path.with_name(path.name + ".pending")
    temporary.write_text(text, encoding=encoding)
    temporary.replace(path)


def normalize_formats(formats, *, allowed=("csv", "tsv", "json")):
    if formats is None:
        return frozenset(allowed)
    values = formats.split(",") if isinstance(formats, str) else list(formats)
    if not values or any(
        not isinstance(v, str) or v.strip() not in allowed for v in values
    ):
        raise ValueError("Supported report formats: " + ",".join(allowed))
    return frozenset(v.strip() for v in values)


def report_policy(value):
    value = {} if value is None else value
    if not isinstance(value, dict) or set(value) - {
        "primary_metric",
        "target_mean_recovery",
    }:
        raise ValueError("Invalid report settings")
    primary = value.get("primary_metric", {})
    if not isinstance(primary, dict) or set(primary) - {
        "classification",
        "detection",
        "face",
        "segmentation",
    }:
        raise ValueError("Invalid report primary_metric mapping")
    if any(not isinstance(v, str) or not v.strip() for v in primary.values()):
        raise ValueError("Report primary metrics require nonempty names")
    target = value.get("target_mean_recovery")
    if target is not None and (
        isinstance(target, bool)
        or not isinstance(target, (float, int))
        or not math.isfinite(target)
        or target <= 0
    ):
        raise ValueError("Target mean recovery must be finite and positive")
    return dict(primary_metric=dict(primary), target_mean_recovery=target)


def export_records(records, directory, *, metadata=None, formats=None):
    """Write selected raw records and tables; omitted formats preserves all outputs."""
    selected = normalize_formats(formats)
    rows, metrics = tables(records)
    raw_rows = [raw_export_row(row) for row in rows]
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, values, fields in (
        ("accuracy", raw_rows, RAW_FIELDS),
        ("results", raw_rows, RAW_FIELDS),
        ("metrics", metrics, METRIC_FIELDS),
    ):
        if "csv" in selected:
            _text(
                directory / (name + ".csv"),
                table_text(values, fields, delimiter=","),
                "utf-8-sig",
            )
        if "tsv" in selected:
            _text(directory / (name + ".tsv"), table_text(values, fields), "utf-8")
    if "json" in selected:
        atomic_json(
            directory / "records.json",
            dict(
                schema_version=SCHEMA_VERSION, metadata=metadata or {}, records=records
            ),
        )
    return rows


def read_records(path):
    path = Path(path)
    if not path.is_file():
        raise ValueError(
            f"Not a measurement records file: {path}. Pass a records.json file, not a folder"
        )
    value = read_json(path)
    if value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported measurement record schema")
    tables(value["records"])
    return value
