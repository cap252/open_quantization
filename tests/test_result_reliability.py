"""F01/F02: exercise actual local model execution, persistence and reporting."""

import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import onnx
from onnx import helper as h, numpy_helper as n
from PIL import Image
import yaml

from opennpu_quant._io import atomic_json, read_json, sha256
from opennpu_quant.configuration import load_config
from opennpu_quant.models.spec import ModelSpec
from opennpu_quant.paths import Paths
from opennpu_quant.runner import run


def fixture(root, *, external=False, host=None):
    root.mkdir(parents=True, exist_ok=True)
    local = root / "local.yaml"
    local.write_text("{}\n")
    paths = Paths.resolve(home=root, local=local)
    directory = paths.models / "toy"
    directory.mkdir(parents=True)
    model = h.make_model(
        h.make_graph(
            [
                h.make_node("Flatten", ["images"], ["flat"], name="flatten"),
                h.make_node(
                    "Gemm", ["flat", "weight", "bias"], ["logits"], name="classifier"
                ),
            ],
            "toy",
            [h.make_tensor_value_info("images", 1, [1, 3, 4, 4])],
            [h.make_tensor_value_info("logits", 1, [1, 6])],
            [
                n.from_array(
                    np.arange(288, dtype=np.float32).reshape(48, 6) / 300, "weight"
                ),
                n.from_array(np.arange(6, dtype=np.float32) / 10, "bias"),
            ],
        ),
        opset_imports=[h.make_opsetid("", 18)],
        ir_version=10,
    )
    options = (
        dict(
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location="weights.bin",
            size_threshold=0,
        )
        if external
        else {}
    )
    onnx.save_model(model, directory / "network.onnx", **options)
    recipe = copy.deepcopy(ModelSpec.load("resnet18").recipe)
    recipe["name"] = "toy"
    recipe["preprocess"].update(resize=4, crop=4)
    recipe["input"]["shape"] = [1, 3, 4, 4]
    if host:
        inp, out, shape = (
            ("images", "prepared", [1, 3, 4, 4])
            if host == "preprocess"
            else ("logits", "scores", [1, 6])
        )
        graph = h.make_model(
            h.make_graph(
                [h.make_node("Add", [inp, "offset"], [out])],
                host,
                [h.make_tensor_value_info(inp, 1, shape)],
                [h.make_tensor_value_info(out, 1, shape)],
                [n.from_array(np.zeros(shape, np.float32), "offset")],
            ),
            opset_imports=[h.make_opsetid("", 18)],
            ir_version=10,
        )
        onnx.save_model(
            graph,
            directory / (host + ".onnx"),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=host + ".bin",
            size_threshold=0,
        )
        recipe["graphs"][host] = host + ".onnx"
        if host == "preprocess":
            recipe["network_input_map"] = {"images": "prepared"}
        else:
            recipe["decode"]["output"] = "scores"
    (directory / "recipe.yaml").write_text(yaml.safe_dump(recipe))
    images = root / "images"
    images.mkdir()
    for i in range(4):
        Image.fromarray(np.full((5, 7, 3), i * 50 + 30, np.uint8)).save(
            images / f"{i}.png"
        )
    atomic_json(
        paths.data / "imagenet/manifest.json",
        dict(
            dataset="imagenet",
            root=str(images),
            expected_samples=2,
            calibration=[dict(id=i, path=f"{i}.png", target=5) for i in range(2)],
            validation=[dict(id=i, path=f"{i}.png", target=5) for i in range(2, 4)],
            annotations={},
            reference_bytes_verified=False,
        ),
    )
    config = load_config()
    config["name"] = "reliability"
    config["models"] = ["toy"]
    config["percentiles"]["toy"] = dict(pot=99.9, float=99.9)
    config["calibration"]["samples"]["imagenet"] = 2
    config["conditions"] = {k: config["conditions"][k] for k in ("fp32", "pot_rtn")}
    return paths, config


class ResultReliabilityTests(unittest.TestCase):
    def test_f01_failed_force_does_not_complete_old_quantized_result(self):
        with tempfile.TemporaryDirectory() as temp:
            paths, config = fixture(Path(temp))
            run(config, paths)
            output = paths.runs / config["name"]
            old = output / "toy/pot_rtn/result.json"
            old_bytes = old.read_bytes()
            network = paths.models / "toy/network.onnx"
            model = onnx.load(network)
            model.graph.initializer[1].CopyFrom(
                n.from_array(np.arange(6, dtype=np.float32) + 1, "bias")
            )
            onnx.save_model(model, network)
            with patch(
                "opennpu_quant.runner.quantize",
                side_effect=RuntimeError("injected failure"),
            ):
                with self.assertRaisesRegex(RuntimeError, "injected failure"):
                    run(config, paths, force=True)
            self.assertEqual(old_bytes, old.read_bytes(), "keep the prior measurement")
            summary = read_json(output / "summary.json")
            row = next(r for r in summary["rows"] if r["condition"] == "pot_rtn")
            self.assertEqual("failed", row["state"])
            self.assertFalse(row["complete"])
            self.assertIsNone(row["recovery_pct"])
            self.assertIsNone(summary["mean_recovery"]["pot_rtn"])

    def test_f01_verified_reuse_records_original_attempt_and_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            paths, config = fixture(Path(temp))
            run(config, paths)
            output = paths.runs / config["name"]
            target = output / "toy/pot_rtn/result.json"
            original = target.read_bytes()
            with patch(
                "opennpu_quant.runner.evaluate",
                side_effect=AssertionError("must reuse"),
            ):
                result = run(config, paths)
            self.assertTrue(all(r["complete"] for r in result["rows"]))
            self.assertEqual(original, target.read_bytes())
            status = read_json(output / "status.json")
            reused = status["models"]["toy"]["conditions"]["pot_rtn"]
            self.assertEqual("reused", reused["state"])
            self.assertNotEqual(status["attempt_id"], reused["measurement_attempt_id"])
            self.assertEqual(sha256(target), reused["result_sha256"])
            self.assertEqual(read_json(target)["identity"], reused["identity"])

    def test_f02_external_network_and_host_data_invalidate_completed_result(self):
        for graph in ("network", "preprocess", "postprocess"):
            with self.subTest(graph=graph), tempfile.TemporaryDirectory() as temp:
                paths, config = fixture(
                    Path(temp),
                    external=graph == "network",
                    host=graph if graph != "network" else None,
                )
                config["conditions"] = {"fp32": {}}
                run(config, paths)
                directory = paths.models / "toy"
                path = directory / (graph + ".onnx")
                header_hash = sha256(path)
                header = onnx.load(path, load_external_data=False)
                tensor = header.graph.initializer[-1]
                ext = {v.key: v.value for v in tensor.external_data}
                with (directory / ext["location"]).open("r+b") as stream:
                    stream.seek(int(ext.get("offset", 0)))
                    stream.write(np.float32(100).tobytes())
                self.assertEqual(header_hash, sha256(path))
                with patch(
                    "opennpu_quant.runner.evaluate",
                    side_effect=AssertionError("not authorized to recompute"),
                ):
                    with self.assertRaisesRegex(ValueError, "changed"):
                        run(config, paths)
                summary = read_json(paths.runs / config["name"] / "summary.json")
                self.assertFalse(summary["rows"][0]["complete"])

    def test_f02_identical_content_relocation_can_reuse(self):
        import shutil

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paths, config = fixture(root / "original", external=True, host="preprocess")
            config["conditions"] = {"fp32": {}}
            run(config, paths)
            moved = root / "moved_models"
            shutil.copytree(paths.models, moved)
            relocated = Paths.resolve(
                home=paths.home, models_dir=moved, local=paths.home / "local.yaml"
            )
            with patch(
                "opennpu_quant.runner.evaluate",
                side_effect=AssertionError("same content"),
            ):
                result = run(config, relocated)
            self.assertTrue(result["rows"][0]["complete"])

    def test_report_refuses_different_source_without_attempt_status(
        self,
    ):
        import shutil
        from opennpu_quant.report import report

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paths, config = fixture(root)
            run(config, paths)
            output = paths.runs / config["name"]
            prior = root / "previous_int8"
            shutil.copytree(output / "toy/pot_rtn", prior)
            network = paths.models / "toy/network.onnx"
            model = onnx.load(network)
            model.graph.initializer[1].CopyFrom(
                n.from_array(np.arange(6, dtype=np.float32) + 1, "bias")
            )
            onnx.save_model(model, network)
            run(config, paths, force=True)
            shutil.copytree(prior, output / "toy/pot_rtn", dirs_exist_ok=True)
            (output / "status.json").unlink()
            row = next(r for r in report(output)["rows"] if r["condition"] == "pot_rtn")
            self.assertIsNotNone(row["value_pct"], "preserve the old measurement")
            self.assertIsNone(row["recovery_pct"])
            self.assertIn("mismatch:source_model_identity", row["pairing_reason"])

    def test_f02_effective_input_mapping_invalidates_reuse(self):
        with tempfile.TemporaryDirectory() as temp:
            paths, config = fixture(Path(temp), host="preprocess")
            config["conditions"] = {"fp32": {}}
            directory = paths.models / "toy"
            (directory / "preprocess.bin").write_bytes(
                np.ones((1, 3, 4, 4), np.float32).tobytes()
            )
            run(config, paths)
            atomic_json(
                directory / "model.json",
                dict(
                    files={p.name: sha256(p) for p in directory.glob("*.onnx")},
                    network_input_map={"images": "images"},
                ),
            )
            with self.assertRaisesRegex(ValueError, "changed"):
                run(config, paths)


if __name__ == "__main__":
    unittest.main()
