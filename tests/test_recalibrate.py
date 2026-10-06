import copy
import importlib.util
from dataclasses import replace
import subprocess
import sys
import yaml

import numpy as np
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import onnx

from opennpu_quant import runner
from opennpu_quant._io import read_json, sha256
from opennpu_quant.configuration import load_config
from tests.test_result_reliability import fixture


def all_conditions(config):
    config["conditions"] = load_config()["conditions"]
    config["adaround"].update(
        steps=2, batch_size=2, window_samples=2, window_steps=2, device="cpu"
    )
    config["output"]["save_qdq_models"] = True


def results(out, config):
    return {c: read_json(out / "toy" / c / "result.json") for c in config["conditions"]}


class RecalibrateTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("torch"), "optional AdaRound extra")
    def test_completed_same_source_recalibrates_and_rebuilds_downstream(self):
        with tempfile.TemporaryDirectory() as temp:
            paths, config = fixture(Path(temp))
            all_conditions(config)
            out = paths.runs / config["name"]
            with patch.object(
                runner, "apply_adaround", wraps=runner.apply_adaround
            ) as initial_ada:
                runner.run(config, paths)
            before = results(out, config)
            source = runner._code_identity()
            models = {
                c: (out / "toy" / c / "model.onnx").read_bytes()
                for c in config["conditions"]
                if c != "fp32"
            }
            with (
                patch.object(
                    runner, "calibrate", wraps=runner.calibrate
                ) as calibration,
                patch.object(runner, "quantize", wraps=runner.quantize) as quantization,
                patch.object(
                    runner, "apply_adaround", wraps=runner.apply_adaround
                ) as adaround,
                patch.object(runner, "evaluate", wraps=runner.evaluate) as evaluation,
            ):
                runner.run(config, paths, recalibrate=True)
            self.assertEqual(source, runner._code_identity())
            self.assertEqual(1, calibration.call_count)
            self.assertEqual(2, quantization.call_count)
            self.assertEqual(2, adaround.call_count)
            self.assertEqual(4, evaluation.call_count)
            after = results(out, config)
            self.assertEqual(before["fp32"], after["fp32"])
            request = read_json(out / "toy" / "recalibration.json")
            self.assertEqual("completed", request["state"])
            for c, record in after.items():
                if c == "fp32":
                    continue
                self.assertNotEqual(before[c]["attempt_id"], record["attempt_id"])
                self.assertEqual(
                    request["generation"], record["calibration_generation"]
                )
                self.assertEqual(
                    request["statistics_identity"], record["calibration_identity"]
                )
                self.assertEqual(before[c]["metrics"], record["metrics"])
                self.assertEqual(
                    models[c], (out / "toy" / c / "model.onnx").read_bytes()
                )
                proof = read_json(out / "toy" / c / "complete.json")
                self.assertEqual(request["generation"], proof["calibration_generation"])
                self.assertEqual(
                    sha256(out / "toy" / c / "result.json"), proof["result_sha256"]
                )
                onnx.checker.check_model(onnx.load(out / "toy" / c / "model.onnx"))
            old_dirs = {x.kwargs["checkpoint_dir"] for x in initial_ada.call_args_list}
            new_dirs = {x.kwargs["checkpoint_dir"] for x in adaround.call_args_list}
            self.assertTrue(old_dirs.isdisjoint(new_dirs))
            self.assertTrue(all(request["generation"] in p.parts for p in new_dirs))
            with (
                patch.object(runner, "calibrate", side_effect=AssertionError("reuse")),
                patch.object(runner, "evaluate", side_effect=AssertionError("reuse")),
            ):
                runner.run(config, paths)
            self.assertEqual(after, results(out, config))

    def test_cli_recalibrate_on_completed_current_source(self):
        with tempfile.TemporaryDirectory() as temp:
            paths, config = fixture(Path(temp))
            path = Path(temp) / "run.yaml"
            path.write_text(yaml.safe_dump(config))
            command = [
                sys.executable,
                "-B",
                "-m",
                "opennpu_quant",
                "run",
                "--home",
                str(paths.home),
                "--config",
                str(path),
            ]
            subprocess.run(command, check=True, capture_output=True, text=True)
            out = paths.runs / config["name"]
            before = results(out, config)
            subprocess.run(
                command + ["--recalibrate"], check=True, capture_output=True, text=True
            )
            after = results(out, config)
            self.assertEqual(before["fp32"], after["fp32"])
            self.assertNotEqual(
                before["pot_rtn"]["attempt_id"], after["pot_rtn"]["attempt_id"]
            )
            self.assertEqual(
                before["pot_rtn"]["identity"], after["pot_rtn"]["identity"]
            )
            self.assertIn("calibration_generation", after["pot_rtn"])

    @unittest.skipUnless(importlib.util.find_spec("torch"), "optional AdaRound extra")
    def test_changed_statistics_drive_new_rtn_adaround_and_evaluation(self):
        from opennpu_quant.configuration import runtime, scheme_for
        from opennpu_quant.data.samples import Dataset
        from opennpu_quant.models.bundle import ModelBundle
        from opennpu_quant.models.spec import ModelSpec
        from opennpu_quant.quant.calibration import statistics_identity
        from opennpu_quant.quant.config import AdaroundConfig
        from opennpu_quant.ort.session import create_session

        with tempfile.TemporaryDirectory() as temp:
            paths, config = fixture(Path(temp))
            all_conditions(config)
            runner.run(config, paths)
            out = paths.runs / config["name"]
            before = results(out, config)
            old_models = {
                c: sha256(out / "toy" / c / "model.onnx")
                for c in config["conditions"]
                if c != "fp32"
            }
            collect = runner.calibrate
            collected = []

            def changed(*args, **kwargs):
                stats = collect(*args, **kwargs)
                hist = copy.deepcopy(stats.histograms)
                # A controlled alternative collector result exercises downstream invalidation.
                for row in hist.rows.values():
                    if row["low"] == row["high"]:
                        continue
                    center = float(row["low"]) + 0.35 * float(row["high"] - row["low"])
                    for key, value in (("signed", center), ("absolute", abs(center))):
                        index = int(
                            np.searchsorted(row[key + "_edges"], value, side="right")
                            - 1
                        )
                        row[key][:] = 0
                        row[key][min(max(index, 0), hist.bins - 1)] = row["count"]
                stats = replace(
                    stats,
                    histograms=hist,
                    statistics_identity=statistics_identity(stats.minmax, hist),
                )
                collected.append(stats)
                return stats

            with patch.object(runner, "calibrate", side_effect=changed):
                runner.run(config, paths, recalibrate=True)
            after = results(out, config)
            self.assertNotEqual(
                before["pot_rtn"]["calibration_identity"],
                after["pot_rtn"]["calibration_identity"],
            )
            bundle = ModelBundle(
                ModelSpec.load(paths.models / "toy/recipe.yaml"), paths.models / "toy"
            )
            dataset = Dataset(paths.data / "imagenet/manifest.json")
            feeds = dataset.feeds(bundle, 2)
            execution = runtime(config)
            samples = list(dataset.samples(bundle))
            for c, policy in config["conditions"].items():
                if not policy:
                    continue
                expected = runner.quantize(
                    bundle.paths["network"],
                    collected[0],
                    scheme_for(config, "toy", policy["scheme"]),
                    ort=execution,
                )
                if policy.get("adaround"):
                    options = dict(config["adaround"])
                    options.pop("data")
                    expected = runner.apply_adaround(
                        bundle.paths["network"],
                        expected,
                        feeds,
                        AdaroundConfig(**options),
                        ort=execution,
                    )
                path = out / "toy" / c / "model.onnx"
                self.assertNotEqual(old_models[c], sha256(path))
                self.assertEqual(
                    expected.model.SerializeToString(),
                    onnx.load(path).SerializeToString(),
                )
                actual_session = create_session(path, execution)
                expected_session = create_session(expected.model, execution)
                for sample in samples:
                    for a, b in zip(
                        actual_session.run(None, sample.feeds),
                        expected_session.run(None, sample.feeds),
                    ):
                        np.testing.assert_array_equal(a, b)
                values = runner.evaluate(
                    expected.model,
                    iter(samples),
                    evaluator=runner.ModelEvaluator(bundle, dataset),
                    ort=execution,
                    expected_samples=2,
                )
                self.assertEqual(values.metrics, after[c]["metrics"])
                self.assertEqual(
                    collected[0].statistics_identity, after[c]["calibration_identity"]
                )

    def test_partial_recalibration_resumes_new_generation_only(self):
        with tempfile.TemporaryDirectory() as temp:
            paths, config = fixture(Path(temp))
            config["conditions"]["float_rtn"] = load_config()["conditions"]["float_rtn"]
            runner.run(config, paths)
            out = paths.runs / config["name"]
            before = results(out, config)
            real = runner.evaluate
            calls = 0

            def interrupted(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise RuntimeError("interrupted second INT8 evaluation")
                return real(*args, **kwargs)

            with patch.object(runner, "evaluate", side_effect=interrupted):
                with self.assertRaisesRegex(RuntimeError, "second INT8"):
                    runner.run(config, paths, recalibrate=True)
            partial = results(out, config)
            self.assertNotEqual(
                before["pot_rtn"]["attempt_id"], partial["pot_rtn"]["attempt_id"]
            )
            self.assertEqual(before["float_rtn"], partial["float_rtn"])
            with (
                patch.object(
                    runner, "calibrate", side_effect=AssertionError("saved new stats")
                ),
                patch.object(runner, "evaluate", wraps=real) as evaluation,
            ):
                runner.run(config, paths)
            self.assertEqual(1, evaluation.call_count)
            after = results(out, config)
            self.assertEqual(partial["pot_rtn"], after["pot_rtn"])
            self.assertNotEqual(
                before["float_rtn"]["attempt_id"], after["float_rtn"]["attempt_id"]
            )
            self.assertEqual(
                after["pot_rtn"]["calibration_generation"],
                after["float_rtn"]["calibration_generation"],
            )
            self.assertEqual(
                after["pot_rtn"]["calibration_identity"],
                after["float_rtn"]["calibration_identity"],
            )

    def test_interrupted_collection_cannot_resume_old_completed_results(self):
        with tempfile.TemporaryDirectory() as temp:
            paths, config = fixture(Path(temp))
            runner.run(config, paths)
            out = paths.runs / config["name"]
            before = results(out, config)
            with patch.object(
                runner, "calibrate", side_effect=RuntimeError("collector interrupted")
            ):
                with self.assertRaisesRegex(RuntimeError, "collector interrupted"):
                    runner.run(config, paths, recalibrate=True)
            with patch.object(
                runner, "calibrate", wraps=runner.calibrate
            ) as calibration:
                runner.run(config, paths)
            self.assertEqual(1, calibration.call_count)
            after = results(out, config)
            self.assertEqual(before["fp32"], after["fp32"])
            self.assertNotEqual(
                before["pot_rtn"]["attempt_id"], after["pot_rtn"]["attempt_id"]
            )

    def test_interruption_invalidates_not_yet_started_models(self):
        with tempfile.TemporaryDirectory() as temp:
            paths, config = fixture(Path(temp))
            shutil.copytree(paths.models / "toy", paths.models / "toy2")
            recipe_path = paths.models / "toy2/recipe.yaml"
            recipe = yaml.safe_load(recipe_path.read_text())
            recipe["name"] = "toy2"
            recipe_path.write_text(yaml.safe_dump(recipe))
            config["models"].append("toy2")
            config["percentiles"]["toy2"] = dict(config["percentiles"]["toy"])
            runner.run(config, paths)
            out = paths.runs / config["name"]
            before = read_json(out / "toy2/pot_rtn/result.json")
            with patch.object(
                runner, "calibrate", side_effect=RuntimeError("first model interrupted")
            ):
                with self.assertRaisesRegex(RuntimeError, "first model interrupted"):
                    runner.run(config, paths, recalibrate=True)
            with patch.object(
                runner, "calibrate", wraps=runner.calibrate
            ) as calibration:
                runner.run(config, paths)
            self.assertEqual(2, calibration.call_count)
            after = read_json(out / "toy2/pot_rtn/result.json")
            self.assertNotEqual(before["attempt_id"], after["attempt_id"])
            self.assertEqual(
                after["calibration_generation"],
                read_json(out / "toy/pot_rtn/result.json")["calibration_generation"],
            )

    def test_reference_rejected_up_front_for_completed_and_partial_runs(self):
        with tempfile.TemporaryDirectory() as temp:
            paths, config = fixture(Path(temp))
            runner.run(config, paths)
            parent = paths.cache / "calibration" / "toy"
            source = next(p for p in parent.iterdir() if (p / "complete.json").exists())
            shutil.copytree(source, parent / "reference")
            reference = copy.deepcopy(config)
            reference["name"] = "reference"
            reference["calibration"]["source"] = "reference"
            runner.run(reference, paths)
            out = paths.runs / reference["name"]
            for partial in (False, True):
                if partial:
                    (out / "toy/pot_rtn/complete.json").unlink()
                before = {
                    str(p.relative_to(out)): sha256(p)
                    for p in out.rglob("*")
                    if p.is_file()
                }
                with patch.object(
                    runner, "calibrate", side_effect=AssertionError("reject first")
                ):
                    with self.assertRaisesRegex(
                        ValueError, "calibration.source=compute"
                    ):
                        runner.run(reference, paths, recalibrate=True)
                after = {
                    str(p.relative_to(out)): sha256(p)
                    for p in out.rglob("*")
                    if p.is_file()
                }
                self.assertEqual(before, after)
            reference["name"] = "never_started"
            with self.assertRaisesRegex(ValueError, "calibration.source=compute"):
                runner.run(reference, paths, recalibrate=True)
            self.assertFalse((paths.runs / reference["name"]).exists())

    def test_force_reference_does_not_resume_pending_compute_request(self):
        with tempfile.TemporaryDirectory() as temp:
            paths, config = fixture(Path(temp))
            runner.run(config, paths)
            parent = paths.cache / "calibration" / "toy"
            source = next(p for p in parent.iterdir() if (p / "complete.json").exists())
            shutil.copytree(source, parent / "reference")
            reference_hash = sha256(parent / "reference/complete.json")
            with patch.object(
                runner, "calibrate", side_effect=RuntimeError("interrupted")
            ):
                with self.assertRaisesRegex(RuntimeError, "interrupted"):
                    runner.run(config, paths, recalibrate=True)
            config["calibration"]["source"] = "reference"
            with patch.object(
                runner,
                "calibrate",
                side_effect=AssertionError("reference is read-only"),
            ):
                runner.run(config, paths, force=True)
                runner.run(config, paths)
            self.assertEqual(reference_hash, sha256(parent / "reference/complete.json"))
            result = read_json(paths.runs / config["name"] / "toy/pot_rtn/result.json")
            self.assertNotIn("calibration_generation", result)

    def test_fp32_only_rejected_before_creating_run(self):
        with tempfile.TemporaryDirectory() as temp:
            paths, config = fixture(Path(temp))
            config["conditions"] = {"fp32": {}}
            with self.assertRaisesRegex(ValueError, "quantized condition"):
                runner.run(config, paths, recalibrate=True)
            self.assertFalse((paths.runs / config["name"]).exists())
