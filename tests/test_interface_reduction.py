import contextlib
import copy
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import onnx
import yaml

from opennpu_quant import (
    CalibrationConfig,
    EvaluationSample,
    OrtConfig,
)
from opennpu_quant.evaluation.segmentation import _ConfusionMatrix
from opennpu_quant.evaluation.metrics import DecodedEvaluator, SegmentationEvaluator
from opennpu_quant.quant.histograms import FixedHistograms
from opennpu_quant.quant.ranges import select_ranges


class SegmentationTests(unittest.TestCase):
    def sample(self):
        return EvaluationSample("image", {}, np.array([[0, 1], [255, 1]]))

    def test_shared_confusion_preserves_counts_and_detailed_metrics(self):
        sample = self.sample()
        prediction = np.array([[0, 0], [99, 1]])
        old = _ConfusionMatrix(3, 255)
        self.assertEqual((3, 2), old.add(prediction, sample.target))
        new = SegmentationEvaluator(output_name="labels", classes=3)
        new.update({"labels": prediction}, sample)
        np.testing.assert_array_equal([[1, 0, 0], [1, 1, 0], [0, 0, 0]], new.confusion)
        np.testing.assert_array_equal(old.matrix, new.confusion)
        self.assertEqual({"mIoU": 0.5}, new.finalize())
        self.assertEqual(old.metrics(), {**new.finalize(), **new.metric_details})
        self.assertEqual(2 / 3, new.metric_details["pixel_accuracy"])
        self.assertEqual([0.5, 0.5, None], new.metric_details["per_class_iou"])
        self.assertEqual(2, new.metric_details["classes_evaluated"])
        self.assertEqual({"mIoU": "fraction"}, new.metric_units)
        new.reset()
        self.assertEqual({}, new.metric_details)
        with self.assertRaisesRegex(ValueError, "No evaluable"):
            new.finalize()

    def test_logits_labels_ignore_policy_and_invalid_inputs(self):
        logits = np.array(
            [[[[2, 2], [1, 0]], [[0, 0], [1, 2]], [[0, 0], [1, 0]]]], np.float32
        )
        evaluator = SegmentationEvaluator(output_name="y", classes=3)
        evaluator.update({"y": logits}, self.sample())
        self.assertEqual({"mIoU": 0.5}, evaluator.finalize())
        for value in (-1, 3, 0.5, float("nan"), float("inf")):
            labels = np.array([[value, 0], [0, 1]])
            with self.subTest(value=value), self.assertRaises(ValueError):
                evaluator.update({"y": labels}, self.sample())
            with self.assertRaises(ValueError):
                _ConfusionMatrix(3, 255).add(labels, self.sample().target)
        for value in (
            logits.repeat(2, axis=0),
            logits[:, :2],
            logits * np.nan,
            np.zeros((3, 3)),
        ):
            with self.assertRaises(ValueError):
                evaluator.update({"y": value}, self.sample())
        ignored = SegmentationEvaluator(output_name="y", classes=3, ignore_label=-1)
        ignored.update(
            {"y": np.array([[0, 999]])}, EvaluationSample("a", {}, np.array([[0, -1]]))
        )
        self.assertEqual({"mIoU": 1.0}, ignored.finalize())
        ignored.reset()
        ignored.update(
            {"y": np.zeros((1, 1))}, EvaluationSample("b", {}, np.array([[-1]]))
        )
        with self.assertRaises(ValueError):
            ignored.finalize()

    def test_decoded_evaluation_details_and_snapshot_isolation(self):
        from opennpu_quant.evaluation.loop import evaluate

        h = onnx.helper
        model = h.make_model(
            h.make_graph(
                [h.make_node("Identity", ["x"], ["y"])],
                "segmentation",
                [h.make_tensor_value_info("x", 1, [1, 3, 2, 2])],
                [h.make_tensor_value_info("y", 1, [1, 3, 2, 2])],
            ),
            opset_imports=[h.make_opsetid("", 17)],
            ir_version=10,
        )
        seen = []

        def decode(outputs, metadata):
            seen.append(metadata["coordinate_policy"])
            return {"labels": outputs["y"].argmax(1)[0]}

        metric = SegmentationEvaluator(output_name="labels", classes=3)
        evaluator = DecodedEvaluator(decode, metric)
        logits = np.array(
            [[[[2, 2], [1, 0]], [[0, 0], [1, 2]], [[0, 0], [1, 0]]]], np.float32
        )
        sample = EvaluationSample(
            "a", {"x": logits}, self.sample().target, {"coordinate_policy": "restored"}
        )
        result = evaluate(
            model,
            [sample],
            evaluator=evaluator,
            ort=OrtConfig.cpu(profile_samples=0),
            expected_samples=1,
        )
        self.assertEqual(["restored"], seen)
        self.assertEqual({"mIoU": 0.5}, result.metrics)
        self.assertTrue(result.complete)
        self.assertEqual(2 / 3, result.metric_details["pixel_accuracy"])
        self.assertEqual({"mIoU": "fraction"}, result.metric_units)
        metric.metric_details["per_class_iou"][0] = 123
        self.assertEqual(0.5, result.metric_details["per_class_iou"][0])
        evaluator.reset()
        self.assertEqual({}, evaluator.metric_details)
        self.assertIsNone(evaluator.prediction)

    def test_model_adapter_forwards_segmentation_details(self):
        from opennpu_quant.models.adapter import ModelEvaluator

        bundle = SimpleNamespace(
            spec=SimpleNamespace(recipe={"decode": {"kind": "segmentation_labels"}})
        )
        evaluator = ModelEvaluator(bundle, None)
        evaluator.metric.update({"labels": np.array([[0, 0], [99, 1]])}, self.sample())
        self.assertEqual({"mIoU": 0.5}, evaluator.finalize())
        self.assertEqual(21, len(evaluator.metric_details["per_class_iou"]))
        self.assertEqual(2, evaluator.metric_details["classes_evaluated"])
        self.assertEqual({"mIoU": "fraction"}, evaluator.metric_units)


class RangeTests(unittest.TestCase):
    def histograms(self):
        ranges = {"x": {"lowest": -4.0, "highest": 12.0}}
        hist = FixedHistograms(ranges, bins=256)
        hist.add("x", np.linspace(-4, 12, 513, dtype=np.float32))
        hist.add("x", np.zeros(51, np.float32))
        return hist, ranges

    def test_range_selection_preserves_statistics(self):
        from opennpu_quant.quant.calibration import statistics_identity

        hist, ranges = self.histograms()
        identity = statistics_identity(ranges, hist)
        with contextlib.redirect_stdout(io.StringIO()):
            for method in ("MinMax", "Percentile", "Entropy"):
                for symmetric in (False, True):
                    config = CalibrationConfig(
                        method,
                        99.9 if method == "Percentile" else None,
                        histogram_bins=256,
                    )
                    value = select_ranges(
                        hist, ranges, config=config, activation_symmetric=symmetric
                    )
                    self.assertEqual({"x"}, set(value))
                    self.assertLessEqual(value["x"]["lowest"], value["x"]["highest"])
                    self.assertTrue(np.isfinite(list(value["x"].values())).all())
                    if method == "Percentile":
                        self.assertGreaterEqual(
                            value["x"]["lowest"], ranges["x"]["lowest"]
                        )
                        self.assertLessEqual(
                            value["x"]["highest"], ranges["x"]["highest"]
                        )
        self.assertEqual(identity, statistics_identity(ranges, hist))
        self.assertIs(
            ranges,
            select_ranges(
                None, ranges, config=CalibrationConfig(), activation_symmetric=False
            ),
        )

    def test_empty_statistics_and_invalid_config(self):
        hist, ranges = self.histograms()
        cfg = CalibrationConfig.percentile(99.9, histogram_bins=256)
        for empty in (None, FixedHistograms({}, 256), FixedHistograms(ranges, 256)):
            with self.assertRaisesRegex(ValueError, "No histogram samples"):
                select_ranges(empty, ranges, config=cfg, activation_symmetric=False)
        with self.assertRaisesRegex(ValueError, "bin count"):
            select_ranges(
                hist,
                ranges,
                config=CalibrationConfig.percentile(),
                activation_symmetric=False,
            )
        with self.assertRaises(ValueError):
            CalibrationConfig.percentile(0)


class InterfaceTests(unittest.TestCase):
    def test_optional_recipe_metadata_and_boundary_validation(self):
        from opennpu_quant.models.spec import ModelSpec, model_names

        for name in model_names():
            recipe = ModelSpec.load(name).recipe
            self.assertNotIn("outputs", recipe)
            self.assertNotIn("network_outputs", recipe)
            old = copy.deepcopy(recipe)
            if "split" in old["export"]:
                old["network_outputs"] = list(old["export"]["split"]["network_outputs"])
            elif old["task"] == "classification":
                old["outputs"] = {"logits": [1, 1000]}
            before = copy.deepcopy(old)
            with tempfile.TemporaryDirectory() as d:
                path = Path(d) / "recipe.yaml"
                path.write_text(yaml.safe_dump(old))
                self.assertEqual(old, ModelSpec.load(path).recipe)
            self.assertEqual(before, old)
        old = copy.deepcopy(ModelSpec.load("yolov5s").recipe)
        old["network_outputs"] = list(
            reversed(old["export"]["split"]["network_outputs"])
        )
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            ModelSpec(old)
        ssd = copy.deepcopy(ModelSpec.load("ssd_mobilenet_v2_320").recipe)
        ssd["network_outputs"] = ["old_exporter_suffix"]
        self.assertEqual(ssd, ModelSpec(ssd).recipe)

    def test_cli_export_check_is_hidden_but_accepted(self):
        from opennpu_quant.cli import main

        with (
            contextlib.redirect_stdout(io.StringIO()) as out,
            self.assertRaises(SystemExit) as done,
        ):
            main(["export", "--help"])
        self.assertEqual(0, done.exception.code)
        self.assertNotIn("--check", out.getvalue())
        with (
            tempfile.TemporaryDirectory() as d,
            patch("opennpu_quant.export.launcher.export", return_value={}) as export,
        ):
            for extra in ([], ["--check"]):
                with contextlib.redirect_stdout(io.StringIO()):
                    main(["export", "resnet18", "--home", d, *extra])
            self.assertEqual(export.call_args_list[0], export.call_args_list[1])
        version = subprocess.check_output(
            [sys.executable, "-B", "-m", "opennpu_quant", "--version"], text=True
        ).strip()
        from opennpu_quant import __version__

        self.assertEqual("opennpu-quant " + __version__, version)
        import tomllib

        metadata = tomllib.loads(
            (Path(__file__).parents[1] / "pyproject.toml").read_text()
        )
        self.assertEqual(metadata["project"]["version"], __version__)

    @unittest.skipUnless(
        importlib.util.find_spec("pycocotools"), "Optional COCO dependency"
    )
    def test_keypoints_remains_available_without_a_packaged_pose_adapter(self):
        from opennpu_quant.evaluation.coco import CocoMetric

        points = [v for i in range(17) for v in (20 + i, 20 + i, 2)]
        annotations = dict(
            info={},
            images=[dict(id=1, width=100, height=100)],
            categories=[
                dict(
                    id=1,
                    name="person",
                    keypoints=[str(i) for i in range(17)],
                    skeleton=[],
                )
            ],
            annotations=[
                dict(
                    id=1,
                    image_id=1,
                    category_id=1,
                    bbox=[10, 10, 80, 80],
                    area=6400,
                    iscrowd=0,
                    keypoints=points,
                    num_keypoints=17,
                )
            ],
        )
        with (
            tempfile.TemporaryDirectory() as d,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            path = Path(d) / "person.json"
            path.write_text(json.dumps(annotations))
            metric = CocoMetric(path, iou_type="keypoints")
            wrapped = DecodedEvaluator(lambda outputs, meta: outputs["points"], metric)
            sample = EvaluationSample("1", {}, None)
            wrapped.update(
                {"points": [dict(category_id=1, keypoints=points, score=1.0)]}, sample
            )
            self.assertAlmostEqual(1.0, wrapped.finalize()["AP"])
            wrapped.reset()
            wrapped.update({"points": []}, sample)
            self.assertEqual(0.0, wrapped.finalize()["AP"])
