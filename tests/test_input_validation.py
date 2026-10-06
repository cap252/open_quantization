"""Review regressions for FP32 denominators and user input boundaries."""

import contextlib
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml

from opennpu_quant.cli import main
from opennpu_quant.configuration import load_config, validate
from opennpu_quant.paths import Paths
from opennpu_quant.results import tables
from tests.test_results import record


class InputValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def invoke(self, *args):
        return subprocess.run(
            [sys.executable, "-m", "opennpu_quant", *map(str, args)],
            cwd=self.root,
            capture_output=True,
            text=True,
        )

    def config_file(self, config):
        path = self.root / "experiment.yaml"
        path.write_text(yaml.safe_dump(config))
        return path

    def assert_error(self, result, *words):
        self.assertEqual(2, result.returncode, result.stdout + result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        for word in words:
            self.assertIn(word, result.stderr)

    def test_fp32_configuration_rejects_quantization(self):
        for row in (
            {"scheme": "pot"},
            {"adaround": False},
            {"scheme": "pot", "adaround": True},
        ):
            config = load_config("quick")
            config["conditions"]["fp32"] = row
            with self.subTest(row=row):
                self.assert_error(
                    self.invoke("run", "-c", self.config_file(config), "--dry-run"),
                    "conditions.fp32",
                    "{}",
                )

    def test_runner_rejects_quantized_fp32_before_creating_run(self):
        from opennpu_quant.runner import run

        config = load_config("quick")
        config["conditions"]["fp32"] = {"scheme": "pot"}
        paths = Paths.resolve(home=self.root)
        with patch("opennpu_quant.runner._run_model") as execute:
            with self.assertRaisesRegex(ValueError, "conditions.fp32"):
                run(config, paths)
            execute.assert_not_called()
        self.assertFalse(paths.runs.exists())

    def test_missing_fp32_in_explicit_selection_is_actionable(self):
        config = load_config("quick")
        del config["conditions"]["fp32"]
        path = self.config_file(config)
        self.assert_error(
            self.invoke("run", "-c", path, "--conditions", "pot_rtn", "--dry-run"),
            "conditions.fp32",
            "{}",
        )
        # Explicit configuration without an FP32 baseline remains usable without
        # CLI selection, which otherwise promises to add an existing baseline.
        self.assertEqual(0, self.invoke("run", "-c", path, "--dry-run").returncode)

    def test_invalid_counts_and_flags_fail_dry_run_with_exact_path(self):
        cases = {
            "runtime.intra_op_threads": [-1, 1.5, True, "1"],
            "runtime.inter_op_threads": [-1, False],
            "runtime.cuda.gpu_mem_limit": [0, -1, 1.5, True],
            "runtime.cuda.use_tf32": ["false", 0],
            "runtime.cuda.cudnn_conv_use_max_workspace": ["false"],
            "evaluation.limit": [0, -1, 1.5, True, "all"],
            "calibration.histogram_bins": [0, 32, -1, 1.5, True],
            "schemes.pot.activation_symmetric": ["false", 0],
            "schemes.pot.quantized_bins": [1, 2.5, True, 4096],
            "percentiles.resnet18.pot": [True, 0, 101, float("nan")],
            "adaround.steps": [0, 1.5, True],
            "adaround.seed": [-1, False],
            "adaround.learning_rate": [0, -1, True, float("inf")],
            "adaround.warmup_fraction": [1, -0.1, "false"],
            "calibration.samples": [{}],
            "calibration.samples.imagenet": [0, -1, 1.5, True, "8"],
            "conditions": [{}],
            "conditions.pot_rtn.adaround": ["false", 0],
            "output.save_predictions": ["false", 0, None],
            "output.save_qdq_models": ["false", 1],
            "activation_cache.enabled": ["false", 0],
            "activation_cache.host_bytes": [-1, 1.5, True],
            "calibration.input_cache.enabled": ["false"],
            "calibration.input_cache.max_bytes": [0],
        }
        # Exercise main directly here to keep the many cases inexpensive; any
        # uncaught exception fails the test rather than being hidden.
        for path, values in cases.items():
            for value in values:
                with self.subTest(path=path, value=value):
                    config = load_config("quick")
                    node = config
                    *parents, key = path.split(".")
                    for parent in parents:
                        node = node[parent]
                    node[key] = value
                    stderr = io.StringIO()
                    with (
                        contextlib.redirect_stdout(io.StringIO()),
                        contextlib.redirect_stderr(stderr),
                    ):
                        with self.assertRaises(SystemExit) as caught:
                            main(
                                [
                                    "run",
                                    "-c",
                                    str(self.config_file(config)),
                                    "--dry-run",
                                ]
                            )
                    self.assertEqual(2, caught.exception.code)
                    self.assertIn(path, stderr.getvalue())
                    self.assertNotIn("Traceback", stderr.getvalue())

    def test_required_packaged_dataset_samples_are_checked(self):
        config = load_config("quick")
        del config["calibration"]["samples"]["imagenet"]
        self.assert_error(
            self.invoke("run", "-c", self.config_file(config), "--dry-run"),
            "calibration.samples.imagenet",
        )

    def test_required_local_dataset_samples_are_checked(self):
        from opennpu_quant.models.spec import ModelSpec

        config = load_config("quick")
        config["models"] = ["local_toy"]
        config["percentiles"]["local_toy"] = config["percentiles"]["resnet18"]
        folder = self.root / "models/local_toy"
        folder.mkdir(parents=True)
        recipe = copy.deepcopy(ModelSpec.load("resnet18").recipe)
        recipe.update(name="local_toy", dataset="custom_data")
        (folder / "recipe.yaml").write_text(yaml.safe_dump(recipe))
        path = self.config_file(config)
        self.assert_error(
            self.invoke("run", "-c", path, "--home", self.root, "--dry-run"),
            "calibration.samples.custom_data",
        )
        config["calibration"]["samples"]["custom_data"] = 2
        result = self.invoke(
            "run", "-c", self.config_file(config), "--home", self.root, "--dry-run"
        )
        self.assertEqual(0, result.returncode, result.stderr)

    def test_local_recipe_can_override_a_packaged_models_dataset(self):
        from opennpu_quant.models.spec import ModelSpec

        config = load_config("quick")
        config["models"] = ["resnet18"]
        config["calibration"]["samples"] = {"custom_data": 2}
        folder = self.root / "models/resnet18"
        folder.mkdir(parents=True)
        recipe = copy.deepcopy(ModelSpec.load("resnet18").recipe)
        recipe["dataset"] = "custom_data"
        (folder / "recipe.yaml").write_text(yaml.safe_dump(recipe))
        result = self.invoke(
            "run", "-c", self.config_file(config), "--home", self.root, "--dry-run"
        )
        self.assertEqual(0, result.returncode, result.stderr)

    def test_valid_zero_threads_unlimited_evaluation_and_false_flags(self):
        config = load_config("quick")
        config["runtime"].update(intra_op_threads=0, inter_op_threads=0)
        config["evaluation"]["limit"] = None
        config["output"].update(save_predictions=False, save_qdq_models=False)
        config["activation_cache"].update(
            enabled=False, host_bytes=0, device_bytes=0, device_reserve_bytes=0
        )
        validate(config)
        result = self.invoke("run", "-c", self.config_file(config), "--dry-run")
        self.assertEqual(0, result.returncode, result.stderr)
        actual = json.loads(result.stdout)["config"]
        self.assertEqual(config, actual)
        self.assertFalse((self.root / "workspace").exists())

    def test_malformed_experiment_and_local_yaml_show_location(self):
        for option in ("-c", "--local-config"):
            with self.subTest(option=option):
                path = self.root / "broken.yaml"
                path.write_text("name: [broken\n")
                result = self.invoke("run", option, path, "--dry-run")
                self.assert_error(result, str(path), "YAML", "line", "column")

    def test_malformed_local_recipe_yaml_shows_location(self):
        folder = self.root / "models/resnet18"
        folder.mkdir(parents=True)
        path = folder / "recipe.yaml"
        path.write_text("name: [broken\n")
        self.assert_error(
            self.invoke(
                "run", "--models", "resnet18", "--home", self.root, "--dry-run"
            ),
            str(path),
            "YAML",
            "line",
            "column",
        )

    def test_existing_comparison_output_is_preserved_and_actionable(self):
        directory = self.root / "snapshot"
        directory.mkdir()
        marker = directory / "keep.txt"
        marker.write_text("prior snapshot")
        result = self.invoke("compare", "catalog.json", "--output", directory)
        self.assert_error(result, "output", "new")
        self.assertEqual("prior snapshot", marker.read_text())
        self.assertEqual([marker], list(directory.iterdir()))

    def test_expected_filesystem_errors_do_not_mask_programming_errors(self):
        for failure, words in (
            (
                PermissionError(13, "Permission denied", "/restricted"),
                ("permission", "/restricted"),
            ),
            (IsADirectoryError(21, "Is a directory", "/folder"), ("file", "/folder")),
            (
                NotADirectoryError(20, "Not a directory", "/file/child"),
                ("directory", "/file/child"),
            ),
        ):
            with self.subTest(failure=type(failure).__name__):
                stderr = io.StringIO()
                with (
                    patch("opennpu_quant.cli.dispatch", side_effect=failure),
                    contextlib.redirect_stderr(stderr),
                ):
                    with self.assertRaises(SystemExit) as caught:
                        main(["results", "records.json"])
                self.assertEqual(2, caught.exception.code)
                for word in words:
                    self.assertIn(word, stderr.getvalue())
        for failure in (KeyError("bug"), TypeError("bug")):
            with patch("opennpu_quant.cli.dispatch", side_effect=failure):
                with self.assertRaises(type(failure)):
                    main(["results", "records.json"])

    def test_explicit_missing_local_config_fails_without_creating_paths(self):
        missing = self.root / "missing.yaml"
        self.assert_error(
            self.invoke("run", "--local-config", missing, "--dry-run"),
            "--local-config",
            str(missing),
        )
        self.assertFalse((self.root / "workspace").exists())
        with self.assertRaisesRegex(FileNotFoundError, "local-config"):
            Paths.resolve(local=missing)

    def test_implicit_missing_local_config_and_existing_empty_file_are_allowed(self):
        self.assertEqual(0, self.invoke("run", "--dry-run").returncode)
        empty = self.root / "empty.yaml"
        empty.write_text("{}\n")
        self.assertEqual(
            0, self.invoke("run", "--local-config", empty, "--dry-run").returncode
        )
        empty.write_text("runs: custom-runs\n")
        result = self.invoke("run", "--local-config", empty, "--dry-run")
        self.assertEqual(
            str(self.root / "custom-runs"), json.loads(result.stdout)["paths"]["runs"]
        )


class BaselineRecordValidationTests(unittest.TestCase):
    def test_quantized_fp32_is_never_a_denominator_even_for_pending_records(self):
        quantized = record("pot_rtn")["quantization"]
        for settings in (quantized, {}):
            with self.subTest(settings=settings):
                base = record(quantization=settings)
                pending = record("pending", state="pending", metrics={}, complete=False)
                rows, _ = tables([base, record("pot_rtn"), pending])
                for row in rows:
                    self.assertIsNone(row["fp32_pct"])
                    self.assertIsNone(row["recovery_pct"])
                    self.assertIsNone(row["delta_pp"])
                    self.assertEqual("mismatch", row["pairing_status"])
                    self.assertIn("baseline", row["pairing_reason"])
                self.assertEqual(80, rows[1]["int8_pct"])

    def test_report_rejects_quantized_fp32_measurements(self):
        from opennpu_quant.report import report
        from opennpu_quant.runner import run
        from tests.test_result_reliability import fixture

        with tempfile.TemporaryDirectory() as folder:
            paths, config = fixture(Path(folder))
            config["conditions"]["fp32"] = {"scheme": "pot"}
            # Emulate a pre-fix run by bypassing only the new input guard.
            # The actual ONNX execution, record persistence and report all run.
            with (
                patch("opennpu_quant.runner.validate"),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                run(config, paths)
            out = paths.runs / config["name"]
            saved = {
                p: p.read_bytes()
                for p in out.rglob("*.json")
                if p.name in ("result.json", "complete.json")
            }
            result = report(out)
            for row in result["rows"]:
                self.assertEqual("mismatch", row["pairing_status"])
                self.assertIsNotNone(row["int8_pct"])
                self.assertIsNone(row["fp32_pct"])
                self.assertIsNone(row["recovery_pct"])
            self.assertEqual(saved, {p: p.read_bytes() for p in saved})

    def test_results_cli_rejects_quantized_baseline_without_altering_records(self):
        from opennpu_quant._io import atomic_json

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "records.json"
            q = record("pot_rtn")
            atomic_json(
                path,
                {
                    "schema_version": 1,
                    "records": [record(quantization=q["quantization"]), q],
                },
            )
            before = path.read_bytes()
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "opennpu_quant",
                    "results",
                    str(path),
                    "--stdout",
                    "csv",
                    "--view",
                    "full",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(0, result.returncode, result.stderr)
            import csv

            rows = list(csv.DictReader(io.StringIO(result.stdout)))
            for row in rows:
                self.assertEqual("", row["recovery_percent"])
            for row in tables(json.loads(before)["records"])[0]:
                self.assertEqual("mismatch", row["pairing_status"])
            self.assertEqual(before, path.read_bytes())


if __name__ == "__main__":
    unittest.main()
