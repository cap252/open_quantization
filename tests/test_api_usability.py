"""Public documentation and existing run-reuse diagnostics, with unchanged contracts."""

import contextlib
import copy
import doctest
import inspect
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import opennpu_quant as api
from opennpu_quant._feeds import feed_digest
from opennpu_quant._io import atomic_json, read_json
from opennpu_quant.configuration import load_config
from opennpu_quant.paths import Paths
from opennpu_quant.runner import run
from tests.library_fixtures import toy_model, feeds


class PublicDocumentationTests(unittest.TestCase):
    def test_public_names_and_signatures_are_preserved(self):
        expected = json.loads(
            (Path(__file__).parent / "fixtures/public_api_signatures.json").read_text()
        )
        self.assertEqual(list(expected), api.__all__)
        actual = {
            name: str(inspect.signature(getattr(api, name))) for name in api.__all__
        }
        self.assertEqual(expected, actual)

    def test_each_public_name_has_contract_and_example(self):
        for name in api.__all__:
            with self.subTest(name=name):
                doc = inspect.getdoc(getattr(api, name)) or ""
                self.assertIn("Examples", doc)
                self.assertTrue("Parameters" in doc or "Attributes" in doc)
                if inspect.isfunction(getattr(api, name)):
                    self.assertIn("Returns", doc)

    def test_documented_examples_execute_with_the_stated_inputs(self):
        model = toy_model()
        result = api.quantize(model, feeds, config=api.QuantizationConfig())
        signatures = [feed_digest(feed) for feed in feeds()]

        def verify_source():
            self.assertEqual(signatures, [feed_digest(feed) for feed in feeds()])

        context = dict(
            model=model,
            feeds=feeds,
            result=result,
            identity={"ordered_feeds": signatures},
            verify_source=verify_source,
        )
        for name in api.__all__:
            with self.subTest(name=name):
                parser = doctest.DocTestParser()
                test = parser.get_doctest(
                    inspect.getdoc(getattr(api, name)) or "",
                    dict(context),
                    name,
                    None,
                    0,
                )
                self.assertGreater(len(test.examples), 0)
                runner = doctest.DocTestRunner()
                output = io.StringIO()
                result_doc = runner.run(test, out=output.write)
                self.assertEqual(0, result_doc.failed, output.getvalue())

    def test_bare_import_remains_light_and_creates_no_files(self):
        code = """
import sys
from pathlib import Path
import opennpu_quant
assert len(opennpu_quant.__all__) == 16
assert not {'numpy', 'onnx', 'onnxruntime', 'torch', 'tensorflow'} & set(sys.modules)
assert not list(Path('.').iterdir())
"""
        with tempfile.TemporaryDirectory() as folder:
            result = subprocess.run(
                [sys.executable, "-c", code], cwd=folder, text=True, capture_output=True
            )
            self.assertEqual(0, result.returncode, result.stderr)

    def test_replay_factory_contract_remains_unchanged(self):
        model = toy_model()
        stats = api.calibrate(model, feeds)
        self.assertEqual(4, stats.sample_count)
        quantized = api.quantize(model, stats, config=api.QuantizationConfig())
        for make_bad in (
            lambda: list(feeds()),
            lambda: tuple(feeds()),
            lambda: feeds(),
        ):
            with self.subTest(kind=type(make_bad()).__name__):
                with self.assertRaisesRegex(TypeError, "factory"):
                    api.calibrate(model, make_bad())
                with self.assertRaisesRegex(TypeError, "factory"):
                    api.apply_adaround(model, quantized, make_bad())


class ReuseGuidanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        local = self.root / "local.yaml"
        local.write_text("{}\n")
        self.paths = Paths.resolve(home=self.root, local=local)
        self.config = load_config("quick")

    def config_conflict(self, before, after):
        out = self.paths.runs / after["name"]
        recorded = out / "effective_config.json"
        atomic_json(recorded, before)
        marker = out / "prior-result.json"
        atomic_json(marker, {"measured": "keep"})
        saved = recorded.read_bytes(), marker.read_bytes()
        with patch("opennpu_quant.runner._run_model") as model_run:
            with self.assertRaises(ValueError) as caught:
                run(after, self.paths)
            model_run.assert_not_called()
        self.assertEqual(saved, (recorded.read_bytes(), marker.read_bytes()))
        self.assertFalse((out / "status.json").exists())
        message = str(caught.exception)
        self.assertIn("--output", message)
        self.assertIn("--force", message)
        self.assertLess(message.index("--output"), message.index("--force"))
        self.assertIn("calibration caches", message)
        self.assertIn("checkpoints may still be reused", message)
        return message

    def test_config_added_removed_and_changed_paths_are_sorted(self):
        before, after = copy.deepcopy(self.config), copy.deepcopy(self.config)
        del before["runtime"]["cuda"]["gpu_mem_limit"]
        del after["activation_cache"]
        after["calibration"]["input_cache"]["enabled"] = False
        message = self.config_conflict(before, after)
        paths = [
            "activation_cache",
            "calibration.input_cache.enabled",
            "runtime.cuda.gpu_mem_limit",
        ]
        self.assertIn("Changed keys: " + ", ".join(paths) + ".", message)
        self.assertNotIn("more)", message)

    def test_changed_keys_are_limited_to_ten_with_remaining_count(self):
        before, after = copy.deepcopy(self.config), copy.deepcopy(self.config)
        updates = {
            "activation_cache.device_bytes": 1024,
            "activation_cache.device_reserve_bytes": 0,
            "activation_cache.host_bytes": 2048,
            "adaround.steps": 20,
            "adaround.batch_size": 2,
            "adaround.window_samples": 4,
            "adaround.window_bytes": 1024,
            "adaround.window_steps": 5,
            "adaround.seed": 2,
            "adaround.learning_rate": 0.002,
            "adaround.regularization": 0.02,
            "adaround.warmup_fraction": 0.3,
        }
        for path, value in updates.items():
            section, name = path.split(".")
            after[section][name] = value
        message = self.config_conflict(before, after)
        paths = sorted(updates)
        self.assertIn("Changed keys: " + ", ".join(paths[:10]) + " (+2 more).", message)
        for path in paths[10:]:
            self.assertNotIn(path, message)

    def test_unchanged_configuration_still_reuses_verified_results(self):
        from tests.test_result_reliability import fixture

        paths, config = fixture(self.root)
        with contextlib.redirect_stdout(io.StringIO()):
            run(config, paths)
        out = paths.runs / config["name"]
        complete = out / "toy/pot_rtn/complete.json"
        original = complete.read_bytes()
        with (
            patch("opennpu_quant.runner.calibrate") as calibration,
            patch("opennpu_quant.runner.quantize") as quantization,
            patch("opennpu_quant.runner.evaluate") as evaluation,
        ):
            run(config, paths)
            calibration.assert_not_called()
            quantization.assert_not_called()
            evaluation.assert_not_called()
        self.assertEqual(original, complete.read_bytes())
        self.assertEqual(
            {"identity", "result_sha256", "model_sha256", "files"},
            set(read_json(complete)),
        )
        self.assertEqual(
            "reused",
            read_json(out / "status.json")["models"]["toy"]["conditions"]["pot_rtn"][
                "state"
            ],
        )

    def test_opaque_identity_conflict_lists_candidates_without_inventing_cause(self):
        from tests.test_result_reliability import fixture

        paths, config = fixture(self.root)
        with contextlib.redirect_stdout(io.StringIO()):
            run(config, paths)
        out = paths.runs / config["name"]
        saved = {
            p: p.read_bytes()
            for p in out.rglob("*.json")
            if p.name in ("complete.json", "result.json")
        }
        with patch(
            "opennpu_quant.runner._code_identity", return_value="different-code"
        ):
            with self.assertRaises(ValueError) as caught:
                run(config, paths)
        message = str(caught.exception)
        for word in (
            "identity",
            "fp32",
            "Possible causes",
            "code",
            "model",
            "inputs",
            "settings",
            "exact cause",
        ):
            self.assertIn(word, message)
        self.assertIn("--output", message)
        self.assertIn("--force", message)
        self.assertLess(message.index("--output"), message.index("--force"))
        self.assertEqual(saved, {p: p.read_bytes() for p in saved})
        self.assertEqual("failed", read_json(out / "status.json")["state"])

    def test_force_keeps_existing_bypass_behavior(self):
        before = copy.deepcopy(self.config)
        before["evaluation"]["limit"] = 1
        out = self.paths.runs / self.config["name"]
        atomic_json(out / "effective_config.json", before)
        with (
            patch("opennpu_quant.runner._run_model") as model_run,
            patch("opennpu_quant.runner.report", return_value={"mean_recovery": {}}),
        ):
            run(self.config, self.paths, force=True)
        self.assertEqual(len(self.config["models"]), model_run.call_count)
        self.assertEqual(self.config, read_json(out / "effective_config.json"))


if __name__ == "__main__":
    unittest.main()
