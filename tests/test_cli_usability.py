"""User-facing help and input errors without changing valid execution contracts."""

import contextlib
import copy
import io
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml

from opennpu_quant.cli import main
from opennpu_quant.configuration import load_config, validate


class CliUsabilityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def invoke(self, args):
        stdout, stderr = io.StringIO(), io.StringIO()
        code = 0
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                main(args)
            except SystemExit as error:
                code = error.code
        return code, stdout.getvalue(), stderr.getvalue()

    def config_file(self, config):
        path = self.root / "experiment.yaml"
        path.write_text(yaml.safe_dump(config))
        return str(path)

    def assert_error(self, args, *words):
        code, _, message = self.invoke(args)
        self.assertEqual(2, code, message)
        self.assertNotIn("Traceback", message)
        for word in words:
            self.assertIn(word, message)
        return message

    def test_run_help_describes_conditions_devices_paths_and_force(self):
        code, message, _ = self.invoke(["run", "--help"])
        self.assertEqual(0, code)
        for word in (
            "fp32",
            "pot_rtn",
            "pot_adaround",
            "float_rtn",
            "float_adaround",
            "cpu",
            "cuda:N",
            "core10",
            "OPENNPU_QUANT_RUNS",
            "CLI",
            "environment",
            "checkpoint",
            "calibration",
            "reuse",
        ):
            self.assertIn(word, message)

    def test_other_commands_explain_inputs_and_first_steps(self):
        for args, words in (
            ([], ["report", "info", "demo", "export", "prepare"]),
            (["info"], ["diagnostics", "OPENNPU_QUANT_HOME"]),
            (["models"], ["model", "recipe"]),
            (["demo"], ["AdaRound", "cuda:N"]),
            (["export"], ["model", "checkpoint"]),
            (["data"], ["prepare", "status"]),
            (["data", "prepare"], ["root", "folder", "hf", "download"]),
            (["data", "status"], ["manifest"]),
            (["fetch"], ["manifest", "sha256", "models"]),
            (["report"], ["run", "folder", "--stdout"]),
            (["results"], ["records.json", "folder", "--stdout"]),
            (["compare"], ["catalog.json", "report", "Excel"]),
        ):
            with self.subTest(command=args):
                code, message, _ = self.invoke([*args, "--help"])
                self.assertEqual(0, code)
                for word in words:
                    self.assertIn(word, message)

    def test_unused_output_is_hidden_but_still_accepted(self):
        for command in (
            ["export", "resnet18"],
            ["data", "prepare", "imagenet", "--root", str(self.root)],
            ["data", "status"],
            ["fetch", "models", "--manifest", "assets.yaml", "--models", "resnet18"],
        ):
            with self.subTest(command=command):
                self.assertNotIn("--output", self.invoke([*command, "--help"])[1])
                with patch("opennpu_quant.cli.dispatch", return_value=None) as dispatch:
                    self.assertEqual(
                        0, self.invoke([*command, "--output", "unused"])[0]
                    )
                self.assertEqual("unused", dispatch.call_args.args[0].output)

    def test_unknown_model_names_packaged_and_local_alternatives(self):
        self.assert_error(
            ["run", "--home", str(self.root), "--models", "resnet", "--dry-run"],
            "resnet",
            "Packaged models",
            "resnet18",
            str(self.root / "models/resnet/recipe.yaml"),
        )
        self.assertFalse((self.root / "runs").exists())

    def test_unknown_conditions_use_the_selected_config(self):
        config = load_config("quick")
        config["conditions"]["my_rtn"] = config["conditions"].pop("pot_rtn")
        message = self.assert_error(
            [
                "run",
                "-c",
                self.config_file(config),
                "--conditions",
                "typo,also_wrong",
                "--dry-run",
            ],
            "typo",
            "also_wrong",
            "Available",
            "my_rtn",
            "float_rtn",
            "fp32",
        )
        self.assertNotIn("pot_adaround", message)

    def test_local_recipe_custom_condition_and_automatic_fp32(self):
        from opennpu_quant.models.spec import ModelSpec

        config = load_config("quick")
        config["models"] = ["local_toy"]
        config["percentiles"]["local_toy"] = config["percentiles"]["resnet18"]
        config["conditions"]["custom"] = {"scheme": "pot"}
        model_dir = self.root / "models/local_toy"
        model_dir.mkdir(parents=True)
        recipe = copy.deepcopy(ModelSpec.load("resnet18").recipe)
        recipe["name"] = "local_toy"
        (model_dir / "recipe.yaml").write_text(yaml.safe_dump(recipe))
        code, output, error = self.invoke(
            [
                "run",
                "-c",
                self.config_file(config),
                "--home",
                str(self.root),
                "--conditions",
                "custom",
                "--device",
                "cuda:12",
                "--dry-run",
            ]
        )
        self.assertEqual(0, code, error)
        actual = json.loads(output)["config"]
        self.assertEqual(["fp32", "custom"], list(actual["conditions"]))
        self.assertEqual("cuda:12", actual["runtime"]["device"])
        self.assertFalse((self.root / "runs").exists())
        self.assertFalse((self.root / "cache").exists())

    def test_missing_percentiles_report_the_key_path(self):
        for missing in ("model", "family"):
            config = load_config("quick")
            if missing == "model":
                del config["percentiles"]["resnet18"]
            else:
                del config["percentiles"]["resnet18"]["pot"]
            with self.subTest(missing=missing):
                self.assert_error(
                    ["run", "-c", self.config_file(config), "--dry-run"],
                    "percentiles.resnet18",
                    "pot" if missing == "family" else "percentiles",
                )

    def test_malformed_settings_report_paths_without_tracebacks(self):
        changes = (
            ("name", None),
            ("models", None),
            ("models", [{}]),
            ("conditions", []),
            ("conditions.fp32", None),
            ("conditions.pot_rtn.scheme", []),
            ("runtime", None),
            ("runtime.cuda", []),
            ("calibration", []),
            ("calibration.samples", None),
            ("calibration.histogram_bins", "many"),
            ("schemes", None),
            ("schemes.pot", []),
            ("schemes.pot.method", []),
            ("percentiles", []),
            ("percentiles.resnet18", []),
            ("percentiles.resnet18.pot", "many"),
            ("adaround", []),
            ("adaround.learning_rate", "fast"),
            ("activation_cache", []),
            ("evaluation", None),
            ("output", []),
            ("report", []),
        )
        for path, value in changes:
            config = load_config("quick")
            keys = path.split(".")
            node = config
            for key in keys[:-1]:
                node = node[key]
            node[keys[-1]] = value
            with self.subTest(path=path):
                self.assert_error(
                    ["run", "-c", self.config_file(config), "--dry-run"],
                    keys[0],
                )
        for path in ("runtime.device", "calibration.source", "evaluation.split"):
            config = load_config("quick")
            section, field = path.split(".")
            del config[section][field]
            with self.subTest(missing=path):
                self.assert_error(
                    ["run", "-c", self.config_file(config), "--dry-run"], path
                )

    def test_invalid_runtime_devices_fail_dry_run(self):
        for device in ("gpu", "cuda", "cuda:-1", "cuda:x", "cuda:1.0"):
            with self.subTest(device=device):
                self.assert_error(
                    ["run", "--models", "resnet18", "--device", device, "--dry-run"],
                    "runtime.device",
                    "cpu",
                    "cuda:N",
                )

    def test_calibration_devices_and_non_string_devices(self):
        for section in ("runtime", "calibration"):
            for device in (None, 1, [], "gpu"):
                config = load_config("quick")
                config[section]["device"] = device
                with self.subTest(section=section, device=device):
                    self.assert_error(
                        ["run", "-c", self.config_file(config), "--dry-run"],
                        section + ".device",
                    )
        for device in ("run", "cpu", "cuda:0", "cuda:12"):
            config = load_config("quick")
            config["calibration"]["device"] = device
            validate(config)

    def test_missing_dataset_suggests_prepare_with_the_actual_data_directory(self):
        from tests.test_result_reliability import fixture

        paths, config = fixture(self.root / "space in home")
        manifest = paths.data / "imagenet/manifest.json"
        manifest.unlink()
        message = self.assert_error(
            ["run", "-c", self.config_file(config), "--home", str(paths.home)],
            "imagenet",
            str(manifest),
            "data prepare",
            "--root",
            "--data-dir",
        )
        command = shlex.split(message.split("Run: ", 1)[1].strip())
        self.assertEqual(str(paths.data), command[command.index("--data-dir") + 1])

    def test_metadata_commands_explain_missing_or_wrong_paths(self):
        file = self.root / "plain.json"
        file.write_text("{}")
        for command, expected in (
            ("report", "run directory"),
            ("results", "records.json"),
            ("compare", "catalog.json"),
        ):
            paths = (
                (self.root / "missing", file)
                if command == "report"
                else (self.root / "missing", self.root)
            )
            for path in paths:
                with self.subTest(command=command, path=str(path)):
                    args = [command, str(path)]
                    if command == "compare":
                        args += ["--output", str(self.root / "output")]
                    self.assert_error(args, str(path), expected)
        self.assertFalse((self.root / "output").exists())

    def test_metadata_path_errors_do_not_import_numerical_runtimes(self):
        code = """
import sys
from opennpu_quant.cli import main
for command in ('report', 'results', 'compare'):
    args = [command, sys.argv[1]]
    if command == 'compare': args += ['--output', sys.argv[1] + '/output']
    try: main(args)
    except SystemExit as error: assert error.code == 2
assert not {'numpy', 'onnx', 'onnxruntime', 'torch'} & set(sys.modules)
"""
        result = subprocess.run(
            [sys.executable, "-c", code, str(self.root / "missing")],
            capture_output=True,
            text=True,
        )
        self.assertEqual(0, result.returncode, result.stderr)

    def test_metadata_files_need_not_have_prescribed_names(self):
        from opennpu_quant._io import atomic_json, sha256
        from opennpu_quant.comparison import read_catalog
        from opennpu_quant.results import read_records

        records = self.root / "my-measurements.json"
        atomic_json(records, dict(schema_version=1, records=[]))
        self.assertEqual([], read_records(records)["records"])
        catalog = self.root / "my-comparison.json"
        atomic_json(
            catalog,
            dict(
                schema_version=1,
                history=dict(path=records.name, sha256=sha256(records)),
            ),
        )
        self.assertEqual([], read_catalog(catalog)[1])

    def test_feed_factory_errors_explain_replay_and_an_example(self):
        from opennpu_quant import calibrate, quantize, apply_adaround
        from tests.library_fixtures import toy_model, feeds

        model = toy_model()
        quantized = quantize(model, feeds)
        for bad in (list(feeds()), tuple(feeds()), iter(feeds())):
            with self.subTest(kind=type(bad).__name__):
                for call in (
                    lambda: calibrate(model, bad),
                    lambda: apply_adaround(model, quantized, bad),
                ):
                    with self.assertRaises(TypeError) as caught:
                        call()
                    message = str(caught.exception)
                    for word in (
                        "factory",
                        "zero-argument",
                        "iterator",
                        "dict",
                        "lambda: iter(feed_list)",
                    ):
                        self.assertIn(word, message)

    def test_input_cache_errors_identify_the_argument_before_writing(self):
        from opennpu_quant.data.feed_cache import prepare_feed_cache

        options = dict(
            factory=lambda: iter([]),
            verify_source=lambda: None,
            identity={},
            expected_samples=1,
            max_bytes=1024,
        )
        for field, bad in (
            ("factory", []),
            ("verify_source", None),
            ("expected_samples", 0),
            ("expected_samples", True),
            ("max_bytes", -1),
            ("max_bytes", "big"),
        ):
            with (
                self.subTest(field=field, bad=bad),
                self.assertRaisesRegex(ValueError, field),
            ):
                prepare_feed_cache(self.root / "cache", **{**options, field: bad})
        self.assertFalse((self.root / "cache").exists())


if __name__ == "__main__":
    unittest.main()
