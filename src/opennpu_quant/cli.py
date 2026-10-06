import argparse, os, json
from . import __version__


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="opennpu-quant",
        description="ONNX QDQ: float/PoT RTN and fixed-encoding AdaRound",
        epilog=(
            "First steps: info -> demo -> export MODEL -> data prepare DATASET "
            "--root DIR -> run --models MODEL --limit 32 -> report "
            "<output>/<experiment name>. Small runs are smoke checks, not reference accuracy."
        ),
    )
    parser.add_argument(
        "--version", action="version", version="opennpu-quant " + __version__
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def paths(p, *, output=False):
        group = p.add_argument_group(
            "Paths",
            "Precedence: CLI > OPENNPU_QUANT_* environment variables > local YAML "
            "> defaults. Directory defaults are relative to home.",
        )
        group.add_argument(
            "--home", help="Workspace root (OPENNPU_QUANT_HOME; default: ./workspace)"
        )
        group.add_argument(
            "--models-dir",
            help="Exported models (OPENNPU_QUANT_MODELS; default: <home>/models)",
        )
        group.add_argument(
            "--data-dir",
            help="Dataset manifests (OPENNPU_QUANT_DATA; default: <home>/data)",
        )
        group.add_argument(
            "--cache-dir",
            help="Calibration, input and download caches (OPENNPU_QUANT_CACHE; default: <home>/cache)",
        )
        group.add_argument(
            "--output",
            help=(
                "Run parent folder; writes <output>/<experiment name> (OPENNPU_QUANT_RUNS; default: <home>/runs)"
                if output
                else argparse.SUPPRESS
            ),
        )
        group.add_argument(
            "--weights-dir",
            help="Original checkpoints and upstream sources (OPENNPU_QUANT_WEIGHTS; default: <home>/weights)",
        )
        group.add_argument(
            "--local-config",
            help="Path settings YAML (default: ./opennpu_quant.local.yaml)",
        )

    info = sub.add_parser(
        "info",
        help="Environment and path diagnostics",
        description="Show environment and path diagnostics without downloading assets.",
    )
    paths(info, output=True)
    demo = sub.add_parser(
        "demo",
        help="CPU demo with no model or dataset download",
        description="Run a tiny synthetic model; scores are smoke checks, not pretrained accuracy.",
    )
    demo.add_argument(
        "--adaround", action="store_true", help="Also run fixed-encoding AdaRound"
    )
    demo.add_argument(
        "--device", default="cpu", help="Execution device: cpu (default) or cuda:N"
    )
    sub.add_parser(
        "models",
        help="List packaged model recipes",
        description="List the packaged model recipe names for export and run.",
    )
    exp = sub.add_parser(
        "export",
        help="Create a model from its pinned original checkpoint",
        description="Export a model from its pinned original checkpoint into the models folder.",
    )
    exp.add_argument(
        "model", help="Model name; list packaged names with 'opennpu-quant models'"
    )
    paths(exp)
    exp.add_argument(
        "--weights", help="Use a local original checkpoint with required checksum"
    )
    exp.add_argument("--source-dir", help="Local pinned upstream checkout")
    exp.add_argument(
        "--python", help="Optional exporter interpreter; default is the current one"
    )
    exp.add_argument(
        "--check",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    data = sub.add_parser(
        "data",
        help="Prepare local metadata, without copying existing images",
        description="Prepare dataset manifests or inspect their status.",
    )
    ds = data.add_subparsers(dest="data_command", required=True)
    prep = ds.add_parser(
        "prepare",
        help="Prepare a dataset manifest",
        description="Prepare metadata referencing existing images without copying them.",
    )
    prep.add_argument(
        "dataset",
        choices=["imagenet", "coco", "widerface", "voc"],
        help="Dataset whose calibration and validation lists will be prepared",
    )
    paths(prep)
    prep.add_argument(
        "--root", required=True, help="Dataset folder containing the existing images"
    )
    prep.add_argument(
        "--source",
        choices=["hf", "folder"],
        default="folder",
        help="folder: local images (default); hf: ImageNet Hugging Face format",
    )
    prep.add_argument(
        "--download",
        action="store_true",
        help="Download ImageNet explicitly; requires --source hf",
    )
    status = ds.add_parser(
        "status",
        help="Show prepared dataset counts",
        description="Show calibration and validation counts from existing dataset manifests.",
    )
    paths(status)
    run = sub.add_parser(
        "run",
        help="Run a packaged or local YAML experiment",
        description="Run prepared models and datasets using a packaged or local YAML experiment; no downloads.",
    )
    paths(run, output=True)
    run.add_argument(
        "-c",
        "--config",
        default="core10",
        help="Packaged experiment (core10 or quick) or YAML path; default: core10",
    )
    run.add_argument(
        "--models",
        help="Comma-separated model names; see 'opennpu-quant models', or supply a local recipe",
    )
    run.add_argument(
        "--conditions",
        help="Comma-separated keys from the selected config; core10: fp32,pot_rtn,pot_adaround,float_rtn,float_adaround. fp32 is included automatically",
    )
    run.add_argument(
        "--device",
        help="Evaluation and AdaRound device: cpu or cuda:N; also calibration when calibration.device=run",
    )
    run.add_argument(
        "--limit",
        type=int,
        help="Maximum evaluation samples for a smoke check, not reference accuracy",
    )
    run.add_argument(
        "--calibration-samples",
        type=int,
        help="Override calibration sample counts for all datasets",
    )
    run.add_argument(
        "--adaround-steps",
        type=int,
        help="Override AdaRound steps per layer (packaged configs: 10000)",
    )
    run.add_argument(
        "--host-cache-gib",
        type=float,
        help="AdaRound activation cache RAM budget in GiB",
    )
    run.add_argument(
        "--gpu-window-mib",
        type=float,
        help="AdaRound activation cache GPU budget in MiB",
    )
    run.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable the intermediate activation cache",
    )
    run.add_argument(
        "--no-input-cache",
        action="store_true",
        help="Decode calibration inputs instead of using the disk cache",
    )
    run.add_argument(
        "--recalibrate",
        action="store_true",
        help="Recollect FP32 calibration and rebuild selected INT8 conditions; requires calibration.source=compute",
    )
    for name, help_text in {
        "force": "Bypass completed-result reuse and recompute selected results; calibration caches and AdaRound checkpoints may still be reused",
        "save-models": "Save a quantized model.onnx for each INT8 condition",
        "save-predictions": "Save per-sample predictions to predictions.jsonl.gz",
        "strict-versions": "Fail unless ONNX Runtime is the verified version 1.20.2",
        "dry-run": "Print effective configuration and paths without running inference",
    }.items():
        run.add_argument("--" + name, action="store_true", help=help_text)
    rep = sub.add_parser(
        "report",
        help="Rebuild tables from a run folder",
        description="Rebuild tables and summaries from an existing run folder without inference.",
    )
    rep.add_argument(
        "run_dir",
        help="Run folder <output>/<experiment name>, containing effective_config.json",
    )
    rep.add_argument("--format", help="Comma-separated md,csv,tsv,json; default: all")
    rep.add_argument(
        "--view",
        choices=["accuracy", "full"],
        default="accuracy",
        help="Output view; accuracy and full both use the 15 raw columns",
    )
    rep.add_argument(
        "--stdout",
        choices=["tsv", "csv"],
        help="Print only the raw table for Excel or redirection",
    )
    records = sub.add_parser(
        "results",
        help="Export measurement records without inference",
        description="Read a portable measurement file such as records.json without inference.",
    )
    records.add_argument(
        "records_json",
        help="Measurement records file, e.g. records.json (not a folder)",
    )
    records.add_argument(
        "--view",
        choices=["accuracy", "full"],
        default="accuracy",
        help="Output view; accuracy and full both use the 15 raw columns",
    )
    records.add_argument(
        "--output", help="Destination folder for CSV, TSV and JSON exports"
    )
    records.add_argument(
        "--stdout",
        choices=["tsv", "csv"],
        help="Print only the raw table for Excel or redirection",
    )
    comparison = sub.add_parser(
        "compare",
        help="Build a portable cohort comparison without inference",
        description="Build comparison sheets from a catalog.json and its referenced history without inference.",
    )
    comparison.add_argument(
        "catalog_json",
        help="Comparison catalog file, e.g. catalog.json (not a run folder)",
    )
    comparison.add_argument(
        "--output", required=True, help="Destination folder for comparison sheets"
    )
    comparison.add_argument(
        "--xlsx",
        action="store_true",
        help="Also write an Excel workbook; requires the .[report] extra",
    )
    fetch = sub.add_parser(
        "fetch",
        help="Optional prebuilt assets with explicit published manifest",
        description="Fetch optional prebuilt assets from an explicit URL and sha256 manifest.",
    )
    paths(fetch)
    fetch.add_argument(
        "kind",
        choices=["models", "calibration"],
        help="Asset kind: model bundles or calibration statistics",
    )
    fetch.add_argument(
        "--manifest",
        required=True,
        help="YAML manifest containing asset URLs and sha256 checksums",
    )
    fetch.add_argument(
        "--models", required=True, help="Comma-separated model names to fetch"
    )
    args = parser.parse_args(argv)
    # Metadata-only commands do not import numerical runtimes.
    if args.command not in ("report", "results", "compare"):
        # Respect deliberate user limits; setup defaults apply before numerical imports.
        for key in [
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "TF_NUM_INTRAOP_THREADS",
            "TF_NUM_INTEROP_THREADS",
        ]:
            os.environ.setdefault(key, "1")
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            import cv2

            cv2.setNumThreads(1)
        except ImportError:
            pass
    try:
        value = dispatch(args)
        if value is not None:
            print(json.dumps(value, ensure_ascii=False, indent=2, default=str))
    except (
        FileExistsError,
        IsADirectoryError,
        NotADirectoryError,
        PermissionError,
    ) as error:
        hint = {
            FileExistsError: "Choose a new --output path; existing files are preserved.",
            IsADirectoryError: "Pass a file path instead of a directory.",
            NotADirectoryError: "Check that each parent path is a directory.",
            PermissionError: "Check read/write permission for the specified path.",
        }[type(error)]
        parser.exit(2, f"{type(error).__name__}: {error}. {hint}\n")
    except (ValueError, RuntimeError, FileNotFoundError, ImportError) as error:
        parser.exit(2, f"{type(error).__name__}: {error}\n")


def dispatch(args):
    from ._io import read_json

    if args.command == "compare":
        from .comparison import write_comparison

        return write_comparison(args.catalog_json, args.output, xlsx=args.xlsx)
    from .paths import Paths

    if args.command == "results":
        from .results import (
            read_records,
            export_records,
            tables,
            table_text,
            RAW_FIELDS,
            raw_export_row,
        )

        payload = read_records(args.records_json)
        rows = tables(payload["records"])[0]
        if args.output:
            export_records(
                payload["records"], args.output, metadata=payload.get("metadata")
            )
        if args.stdout:
            print(
                table_text(
                    [raw_export_row(row) for row in rows],
                    RAW_FIELDS,
                    delimiter="\t" if args.stdout == "tsv" else ",",
                ),
                end="",
            )
            return None
        metadata = payload.get("metadata", {})
        return dict(
            records=len(rows),
            completed=sum(r["complete"] for r in rows),
            output=args.output,
            source_run=metadata.get("source_run"),
            captured_at=metadata.get("captured_at"),
        )
    if args.command == "models":
        from .models.spec import model_names

        return model_names()
    if args.command == "demo":
        from .demo import run

        return run(adaround=args.adaround, device=args.device)
    if args.command == "report":
        from .report import report

        result = report(args.run_dir, formats=args.format)
        if args.stdout:
            from .results import table_text, RAW_FIELDS, raw_export_row

            print(
                table_text(
                    [raw_export_row(row) for row in result["rows"]],
                    RAW_FIELDS,
                    delimiter="\t" if args.stdout == "tsv" else ",",
                ),
                end="",
            )
            return None
        return result["mean_recovery"]
    keys = ["home", "models_dir", "data_dir", "cache_dir", "output", "weights_dir"]
    paths = Paths.resolve(
        **{k: getattr(args, k, None) for k in keys},
        local=getattr(args, "local_config", None),
    )
    if args.command == "info":
        from .envcheck import info

        return dict(**info(), paths=paths.to_dict())
    if args.command == "data":
        if args.data_command == "status":
            return {
                name: dict(
                    calibration=len(v["calibration"]),
                    validation=len(v["validation"]),
                    root=v["root"],
                )
                for name in ["imagenet", "coco", "widerface", "voc"]
                if (p := paths.data / name / "manifest.json").exists()
                for v in [read_json(p)]
            }
        from .data.prepare import prepare

        v = prepare(
            args.dataset,
            args.root,
            paths.data / args.dataset / "manifest.json",
            source=args.source,
            download=args.download,
        )
        return dict(
            manifest=str(paths.data / args.dataset / "manifest.json"),
            validation=len(v["validation"]),
            calibration=len(v["calibration"]),
        )
    if args.command == "export":
        from .export.launcher import export

        return export(
            args.model,
            paths,
            weights=args.weights,
            source=args.source_dir,
            python=args.python,
        )
    if args.command == "fetch":
        from .fetch import fetch_assets

        return fetch_assets(args.kind, args.manifest, args.models.split(","), paths)
    from .configuration import load_config, validate

    config = load_config(args.config)
    if args.models:
        config["models"] = args.models.split(",")
    from .models.spec import model_names

    packaged = model_names()
    for name in config["models"]:
        local = paths.models / name / "recipe.yaml"
        if name not in packaged and not local.is_file():
            raise ValueError(
                f"Unknown model '{name}'. Packaged models: {', '.join(packaged)}; "
                f"a local model needs {local}"
            )
    if args.conditions:
        chosen = args.conditions.split(",")
        unknown = [n for n in chosen if n not in config["conditions"]]
        if unknown:
            raise ValueError(
                f"Unknown condition(s): {', '.join(unknown)}. "
                f"Available: {', '.join(config['conditions'])}"
            )
        if "fp32" not in chosen:
            if "fp32" not in config["conditions"]:
                raise ValueError(
                    "conditions.fp32 is required for --conditions to include the baseline. Add fp32: {} to conditions."
                )
            chosen.insert(0, "fp32")
        config["conditions"] = {k: config["conditions"][k] for k in chosen}
    if args.device:
        config["runtime"]["device"] = args.device
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("Positive limit required")
        config["evaluation"]["limit"] = args.limit
    if args.calibration_samples is not None:
        if args.calibration_samples < 1:
            raise ValueError("Positive calibration count required")
        config["calibration"]["samples"] = {
            k: args.calibration_samples for k in config["calibration"]["samples"]
        }
    if args.adaround_steps is not None:
        config["adaround"]["steps"] = args.adaround_steps
    if args.save_models:
        config["output"]["save_qdq_models"] = True
    if args.save_predictions:
        config["output"]["save_predictions"] = True
    cc = config.setdefault("activation_cache", {})
    if args.host_cache_gib is not None:
        cc["host_bytes"] = int(args.host_cache_gib * 1024**3)
    if args.gpu_window_mib is not None:
        cc["device_bytes"] = int(args.gpu_window_mib * 1024**2)
    if args.no_cache:
        cc["enabled"] = False
    if args.no_input_cache:
        config["calibration"].setdefault("input_cache", {})["enabled"] = False
    validate(config, models_dir=paths.models)
    if args.strict_versions:
        from .ort.session import runtime
        from .ort.config import OrtConfig

        runtime(OrtConfig.cpu(strict_versions=True))
    if args.dry_run:
        return dict(config=config, paths=paths.to_dict(), will_download=False)
    from .runner import run

    result = run(config, paths, force=args.force, recalibrate=args.recalibrate)
    return dict(
        output=str(paths.runs / config["name"]), mean_recovery=result["mean_recovery"]
    )
