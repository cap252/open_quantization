import copy
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import onnx
from onnx import helper as h, numpy_helper as n
from PIL import Image
import yaml

from opennpu_quant import runner
from opennpu_quant._io import atomic_json, read_json, sha256
from opennpu_quant.data.samples import Dataset
from opennpu_quant.evaluation.metrics import SegmentationEvaluator
from opennpu_quant.models.bundle import ModelBundle
from opennpu_quant.models.spec import ModelSpec
from tests.test_result_reliability import fixture


class SegmentationPredictionTests(unittest.TestCase):
    def test_actual_writer_roundtrip_confusion_metrics_and_save_toggle(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paths, config = fixture(root)
            directory = paths.models / "toy"
            recipe = copy.deepcopy(ModelSpec.load("deeplabv3_mnv2_voc").recipe)
            recipe["name"] = "toy"
            recipe["graphs"] = {"network": "network.onnx"}
            recipe["input"].update(name="images", shape=[1, 3, 8, 8], layout="NCHW")
            recipe.pop("network_input_map", None)
            recipe["preprocess"]["size"] = 8
            recipe["decode"]["output"] = "logits"
            (directory / "recipe.yaml").write_text(yaml.safe_dump(recipe))
            weights = np.arange(63, dtype=np.float32).reshape(21, 3, 1, 1) / 63
            bias = np.linspace(-0.4, 0.2, 21, dtype=np.float32)
            model = h.make_model(
                h.make_graph(
                    [h.make_node("Conv", ["images", "weight", "bias"], ["logits"])],
                    "seg",
                    [h.make_tensor_value_info("images", 1, [1, 3, 8, 8])],
                    [h.make_tensor_value_info("logits", 1, [1, 21, 8, 8])],
                    [n.from_array(weights, "weight"), n.from_array(bias, "bias")],
                ),
                opset_imports=[h.make_opsetid("", 18)],
                ir_version=10,
            )
            onnx.save(model, directory / "network.onnx")
            rows = []
            for i, (height, width) in enumerate(((5, 7), (9, 3))):
                pixels = (
                    np.arange(height * width * 3)
                    .reshape(height, width, 3)
                    .astype(np.uint8)
                )
                Image.fromarray(pixels).save(root / "images" / f"seg{i}.png")
                mask = (np.indices((height, width)).sum(0) % 3 * 10).astype(np.uint8)
                mask[0, 0] = 255
                Image.fromarray(mask).save(root / "images" / f"mask{i}.png")
                rows.append(
                    dict(id=f"seg{i}", path=f"seg{i}.png", target=f"mask{i}.png")
                )
            atomic_json(
                paths.data / "voc/manifest.json",
                dict(
                    dataset="voc",
                    root=str(root / "images"),
                    calibration=rows[:1],
                    validation=rows,
                    expected_samples=2,
                    annotations={},
                    reference_bytes_verified=False,
                ),
            )
            config["conditions"] = {"fp32": {}}
            config["calibration"]["samples"]["voc"] = 1
            captured = []
            actual = runner.ModelEvaluator

            def capture(*args):
                evaluator = actual(*args)
                captured.append(evaluator)
                return evaluator

            records = []
            matrices = []
            for enabled in (False, True):
                config["name"] = "save_on" if enabled else "save_off"
                config["output"]["save_predictions"] = enabled
                with patch.object(runner, "ModelEvaluator", side_effect=capture):
                    runner.run(config, paths)
                out = paths.runs / config["name"] / "toy/fp32"
                records.append(read_json(out / "result.json"))
                matrices.append(captured[-1].metric.confusion.copy())
                self.assertEqual(enabled, (out / "predictions.jsonl.gz").exists())
            self.assertEqual(records[0]["metrics"], records[1]["metrics"])
            self.assertEqual(records[0]["metric_details"], records[1]["metric_details"])
            np.testing.assert_array_equal(*matrices)
            saved = out / "predictions.jsonl.gz"
            self.assertEqual(
                sha256(saved), read_json(out / "complete.json")["files"][saved.name]
            )
            self.assertFalse((out / "predictions.jsonl.gz.pending").exists())
            with gzip.open(saved, "rt") as stream:
                predictions = [json.loads(line) for line in stream]
            self.assertEqual(["seg0", "seg1"], [r["id"] for r in predictions])
            bundle = ModelBundle(ModelSpec.load(directory / "recipe.yaml"), directory)
            dataset = Dataset(paths.data / "voc/manifest.json")
            replay = SegmentationEvaluator(output_name="labels")
            for row, sample in zip(predictions, dataset.samples(bundle)):
                self.assertIsNotNone(
                    row["prediction"], "segmentation must persist recoverable labels"
                )
                labels = np.asarray(row["prediction"]["labels"])
                self.assertEqual(sample.target.shape, labels.shape)
                self.assertTrue(np.issubdtype(labels.dtype, np.integer))
                self.assertTrue(((labels >= 0) & (labels < 21)).all())
                replay.update({"labels": labels}, sample)
            np.testing.assert_array_equal(matrices[1], replay.confusion)
            self.assertEqual(records[1]["metrics"], replay.finalize())
            self.assertEqual(records[1]["metric_details"], replay.metric_details)
