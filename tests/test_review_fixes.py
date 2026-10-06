import copy
import contextlib
import importlib.util
import io
import itertools
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from onnx import helper as h, numpy_helper as n

from opennpu_quant._io import atomic_json, object_hash, read_json, sha256
from opennpu_quant.report import report
from opennpu_quant.results import read_records


class ReportPolicyTests(unittest.TestCase):
    def fixture(self, root, policy=None):
        config = dict(
            name="review",
            models=["m"],
            conditions={"fp32": {}, "int8": {"scheme": "float"}},
            schemes={
                "float": dict(
                    scope="all",
                    activation_symmetric=False,
                    weight_granularity="per_channel",
                    method="percentile",
                    percentile=99.99,
                )
            },
            adaround={},
            evaluation={"split": "validation"},
        )
        if policy is not None:
            config["report"] = policy
        atomic_json(root / "effective_config.json", config)
        for condition, values in (
            ("fp32", {"top1": 0.8, "top5": 0.95}),
            ("int8", {"top1": 0.78, "top5": 0.945}),
        ):
            row = dict(
                model="m",
                title="Model",
                metric="top1",
                metrics=values,
                metric_units={"top1": "fraction", "top5": "fraction"},
                samples=100,
                expected_samples=100,
                complete=True,
                model_identity="source",
                preprocessing_identity="pre",
                evaluation_identity="same",
                config_identity=object_hash(config),
                model_metadata={
                    "task": "classification",
                    "dataset": "imagenet",
                    "metric": "top1",
                    "source": {},
                },
                scheme=None if condition == "fp32" else config["schemes"]["float"],
            )
            self.write(root, condition, row)
        return config

    def write(self, root, condition, row):
        path = root / "m" / condition / "result.json"
        atomic_json(path, row)
        atomic_json(
            path.parent / "complete.json", dict(result_sha256=sha256(path), files={})
        )

    def test_all_format_subsets_and_preserving_unselected_files(self):
        outputs = {
            "md": {"summary.md"},
            "json": {"records.json", "summary.json"},
            "csv": {n + ".csv" for n in ("accuracy", "results", "metrics")},
            "tsv": {n + ".tsv" for n in ("accuracy", "results", "metrics")},
        }
        for count in range(1, 5):
            for chosen in itertools.combinations(outputs, count):
                with self.subTest(chosen=chosen), tempfile.TemporaryDirectory() as d:
                    root = Path(d)
                    self.fixture(root)
                    report(root, formats=chosen)
                    expected = set().union(*(outputs[c] for c in chosen))
                    self.assertEqual(
                        expected,
                        {p.name for p in root.iterdir() if p.is_file()}
                        - {"effective_config.json"},
                    )
                    for p in expected:
                        before = (root / p).read_bytes()
                        report(root, formats="md" if not p.endswith(".md") else "csv")
                        self.assertEqual(before, (root / p).read_bytes())
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.fixture(root)
            report(root)
            self.assertTrue(
                set().union(*outputs.values()) <= {p.name for p in root.iterdir()}
            )

    def test_primary_metric_target_and_source_immutability(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.fixture(
                root,
                dict(
                    primary_metric={"classification": "top5"}, target_mean_recovery=98
                ),
            )
            inputs = {p: p.read_bytes() for p in root.rglob("*.json")}
            value = report(root)
            self.assertTrue(value["target_met"]["int8"])
            self.assertAlmostEqual(100 * 0.945 / 0.95, value["mean_recovery"]["int8"])
            self.assertEqual("top5", value["rows"][1]["metric"])
            self.assertEqual({p: p.read_bytes() for p in inputs}, inputs)
            self.assertEqual(
                value["report_policy"],
                read_records(root / "records.json")["metadata"]["report_policy"],
            )
            row = read_json(root / "m/int8/result.json")
            row["metrics"]["top5"] = 94.5
            row["metric_units"]["top5"] = "percent"
            self.write(root, "int8", row)
            self.assertAlmostEqual(
                value["mean_recovery"]["int8"], report(root)["mean_recovery"]["int8"]
            )

    def test_default_primary_target_and_partial_or_missing_denominator(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.fixture(root, dict(target_mean_recovery=98))
            self.assertFalse(report(root)["target_met"]["int8"])
            self.assertEqual("top1", report(root)["rows"][1]["metric"])
            base = read_json(root / "m/fp32/result.json")
            for update in (
                {"complete": False},
                {"expected_samples": 101},
                {"evaluation_identity": "different"},
                {"metrics": {"top1": 0, "top5": 0}},
            ):
                with self.subTest(update=update):
                    self.write(root, "fp32", dict(base, **update))
                    self.assertIsNone(report(root)["target_met"]["int8"])
            (root / "m/fp32/complete.json").unlink()
            self.assertIsNone(report(root)["mean_recovery"]["int8"])

    def test_bad_policy_missing_metric_and_units_fail_before_writing(self):
        policies = [
            {"target_mean_recovery": v} for v in (True, 0, float("nan"), -1, "98")
        ]
        policies += [
            {"primary_metric": {"classification": "missing"}},
            {"primary_metric": {"oops": "top1"}},
            {"primary_metric": []},
        ]
        for policy in policies:
            with self.subTest(policy=policy), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                self.fixture(root, policy)
                with self.assertRaises(ValueError):
                    report(root)
                self.assertFalse((root / "records.json").exists())
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.fixture(root)
            row = read_json(root / "m/int8/result.json")
            row["metric_units"]["top1"] = "dB"
            self.write(root, "int8", row)
            with self.assertRaisesRegex(ValueError, "incompatible"):
                report(root)
            for formats in ("", "csv,", "pdf", []):
                with self.assertRaises(ValueError):
                    report(root, formats=formats)

    def test_cli_format_and_stdout(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.fixture(root)
            proc = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "opennpu_quant",
                    "report",
                    str(root),
                    "--format",
                    "tsv",
                    "--stdout",
                    "tsv",
                ],
                text=True,
                capture_output=True,
                check=True,
            )
            self.assertEqual((root / "accuracy.tsv").read_text(), proc.stdout)
            self.assertEqual("", proc.stderr)
            self.assertFalse((root / "summary.json").exists())


class ValidationFixTests(unittest.TestCase):
    def test_top5_checks_strict_success_and_exception_paths(self):
        from opennpu_quant.export.finalize import framework_compare
        from opennpu_quant.models.spec import ModelSpec

        recipe = ModelSpec.load("resnet18").recipe
        a = np.array([[0, 1, 2, 3, 4, 4.000001]], np.float32)
        b = a.copy()
        b[0, -2:] = b[0, -2:][::-1]
        np.testing.assert_allclose(a, b, atol=1e-5, rtol=1e-4)
        with self.assertRaisesRegex(ValueError, "Top5"):
            framework_compare({"logits": a}, {"logits": b}, recipe)
        b[0, 0] = 5e-5
        with self.assertRaisesRegex(ValueError, "Top5"):
            framework_compare({"logits": a}, {"logits": b}, recipe)
        self.assertTrue(
            framework_compare({"logits": a}, {"logits": a.copy()}, recipe)[
                "strict_passed"
            ]
        )

    def test_fixed_recipe_settings_are_checked(self):
        from opennpu_quant.models.spec import ModelSpec, model_names

        for name in model_names():
            ModelSpec.load(name)
        cases = [
            ("resnet18", ("export", "framework_parity", "rtol"), 0.01),
            ("resnet18", ("export", "framework_parity", "require_top5_order"), False),
            ("resnet18", ("input", "dtype"), "float16"),
            ("ssd_mobilenet_v2_320", ("preprocess", "interpolation"), "bicubic"),
            ("ssd_mobilenet_v2_320", ("decode", "boxes_to"), "xyxy"),
            ("deeplabv3_mnv2_voc", ("metric_params", "ignore_label"), 0),
            ("deeplabv3_mnv2_voc", ("preprocess", "pad"), "center"),
            ("retinaface_mnet025_640", ("decode", "priors", "clip"), True),
            ("retinaface_mnet025_640", ("metric_params", "thresholds"), 10),
            ("yolov5s", ("decode", "nms", "per_class"), False),
        ]
        for name, path, value in cases:
            with self.subTest(name=name, path=path):
                recipe = copy.deepcopy(ModelSpec.load(name).recipe)
                target = recipe
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = value
                with self.assertRaises(ValueError):
                    ModelSpec(recipe)

    def test_audit_requires_identity_and_detects_block_size_change(self):
        from opennpu_quant.quant.engine import audit_runtime
        from opennpu_quant.quant.parameters import parameter_identity, verify_parameters

        with self.assertRaisesRegex(ValueError, "parameter identity"):
            audit_runtime("unused", "unused", {})
        m = h.make_model(
            h.make_graph(
                [
                    h.make_node(
                        "DequantizeLinear", ["w", "s", "z"], ["out"], name="dq", axis=0
                    )
                ],
                "g",
                [],
                [h.make_tensor_value_info("out", 1, [1])],
                [
                    n.from_array(np.array([1], np.int8), "w"),
                    n.from_array(np.array([0.1], np.float32), "s"),
                    n.from_array(np.array([0], np.int8), "z"),
                ],
            )
        )
        audit = {"parameter_identity": parameter_identity(m)}
        m.graph.node[0].attribute.append(h.make_attribute("block_size", 64))
        with self.assertRaisesRegex(ValueError, "identity changed"):
            verify_parameters(m, audit)

    def test_gate_provenance_and_public_roundtrip_stay_available(self):
        from opennpu_quant import Scheme, OrtConfig
        from opennpu_quant.graph.activations import recognize

        m = h.make_model(
            h.make_graph(
                [
                    h.make_node("Sigmoid", ["b"], ["s"]),
                    h.make_node("Mul", ["a", "s"], ["y"]),
                ],
                "g",
                [h.make_tensor_value_info(k, 1, [1, 4]) for k in ("a", "b")],
                [h.make_tensor_value_info("y", 1, [1, 4])],
            ),
            opset_imports=[h.make_opsetid("", 17)],
        )
        self.assertFalse(recognize(m)["passed"])
        self.assertTrue(recognize(m, {"y": "verified projections"})["passed"])
        self.assertEqual(Scheme(), Scheme.from_dict(Scheme().to_dict()))
        self.assertTrue(Scheme().name)
        self.assertEqual(
            ("CPUExecutionProvider",), OrtConfig.cpu().to_dict()["providers"]
        )

    def test_asset_download_install_is_atomic_and_rejects_replacement(self):
        import tarfile
        import yaml
        from unittest.mock import patch
        from opennpu_quant.fetch import fetch_assets
        from opennpu_quant.paths import Paths

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            archive = root / "bundle.tar.gz"
            with tarfile.open(archive, "w:gz") as tar:
                item = tarfile.TarInfo("network.onnx")
                item.size = 5
                tar.addfile(item, io.BytesIO(b"model"))
            manifest = root / "assets.yaml"
            manifest.write_text(
                yaml.safe_dump(
                    {
                        "models": {
                            "toy": {"url": archive.as_uri(), "sha256": sha256(archive)}
                        }
                    }
                )
            )
            paths = Paths.resolve(home=root / "workspace")
            installed = fetch_assets("models", manifest, ["toy"], paths)
            self.assertEqual(
                b"model", (Path(installed[0]) / "network.onnx").read_bytes()
            )
            with self.assertRaisesRegex(ValueError, "already exists"):
                fetch_assets("models", manifest, ["toy"], paths)
            self.assertEqual(
                b"model", (Path(installed[0]) / "network.onnx").read_bytes()
            )
            second = Paths.resolve(home=root / "interrupted")
            with patch(
                "opennpu_quant.fetch.extract_archive",
                side_effect=ValueError("broken archive"),
            ):
                with self.assertRaisesRegex(ValueError, "broken archive"):
                    fetch_assets("models", manifest, ["toy"], second)
            self.assertFalse((second.models / "toy").exists())
            self.assertFalse(list(second.models.glob(".asset_*")))

    def test_optional_torch_cache_suite(self):
        code = """import importlib.abc, importlib.util, sys, unittest
class NoTorch(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  if fullname == 'torch' or fullname.startswith('torch.'):
   raise ModuleNotFoundError('Torch intentionally unavailable')
original=importlib.util.find_spec
importlib.util.find_spec=lambda name, package=None: None if name=='torch' else original(name,package)
sys.meta_path.insert(0,NoTorch())
suite=unittest.defaultTestLoader.loadTestsFromName('tests.ort.test_adaround_cache')
result=unittest.TextTestRunner().run(suite)
assert result.wasSuccessful()
assert len(result.skipped)==5 and result.testsRun==7
assert 'torch' not in sys.modules
"""
        subprocess.run(
            [sys.executable, "-B", "-c", code],
            capture_output=True,
            text=True,
            check=True,
        )

    @unittest.skipUnless(
        importlib.util.find_spec("pycocotools"), "Optional COCO dependency"
    )
    def test_real_coco_details_reach_evaluation_result(self):
        from opennpu_quant import evaluate, EvaluationSample, OrtConfig
        from opennpu_quant.models.adapter import ModelEvaluator

        with tempfile.TemporaryDirectory() as d:
            annotation = Path(d) / "gt.json"
            atomic_json(
                annotation,
                dict(
                    images=[dict(id=1, width=100, height=100)],
                    categories=[dict(id=1, name="object")],
                    annotations=[
                        dict(
                            id=1,
                            image_id=1,
                            category_id=1,
                            bbox=[10, 10, 20, 20],
                            area=400,
                            iscrowd=0,
                        )
                    ],
                ),
            )
            recipe = dict(
                decode=dict(
                    kind="tf_od_api",
                    count="count",
                    boxes="boxes",
                    scores="scores",
                    classes="classes",
                )
            )
            outputs = {
                "count": np.array([1]),
                "boxes": np.array([[[0.1, 0.1, 0.3, 0.3]]]),
                "scores": np.array([[0.9]]),
                "classes": np.array([[1]]),
            }
            bundle = SimpleNamespace(
                spec=SimpleNamespace(recipe=recipe), finish=lambda out, meta: outputs
            )
            evaluator = ModelEvaluator(
                bundle, SimpleNamespace(annotations={"coco": annotation})
            )
            model = h.make_model(
                h.make_graph(
                    [h.make_node("Identity", ["x"], ["y"])],
                    "g",
                    [h.make_tensor_value_info("x", 1, [1])],
                    [h.make_tensor_value_info("y", 1, [1])],
                ),
                opset_imports=[h.make_opsetid("", 17)],
                ir_version=10,
            )
            sample = EvaluationSample(
                "1", {"x": np.ones(1, np.float32)}, None, {"original_size": [100, 100]}
            )
            with contextlib.redirect_stdout(io.StringIO()):
                result = evaluate(
                    model,
                    [sample],
                    evaluator=evaluator,
                    ort=OrtConfig.cpu(profile_samples=0),
                    expected_samples=1,
                )
            self.assertAlmostEqual(1, result.metrics["AP"])
            self.assertEqual(1, result.metric_details["per_class"][0]["category_id"])
            self.assertEqual("fraction", result.metric_units["AP"])
            evaluator.reset()
            self.assertEqual(1, result.metric_details["image_count"])


if __name__ == "__main__":
    unittest.main()
