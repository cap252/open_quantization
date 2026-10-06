import copy, shutil, subprocess, sys, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as n
from opennpu_quant import *
from opennpu_quant.configuration import load_config
from opennpu_quant._io import atomic_json, read_json


class PortableTests(unittest.TestCase):
    def test_import_does_not_load_numerical_runtimes(self):
        code = "import sys; import opennpu_quant; assert not set(['numpy','onnx','onnxruntime','torch','tensorflow']) & set(sys.modules)"
        subprocess.run([sys.executable, "-c", code], check=True)

    def test_recipes_and_strict_config(self):
        from opennpu_quant.models.spec import model_names, ModelSpec
        from opennpu_quant.configuration import validate

        self.assertEqual(10, len(model_names()))
        for name in model_names():
            self.assertEqual(name, ModelSpec.load(name).name)
        cfg = load_config()
        cfg["runtime"]["typo"] = 1
        with self.assertRaises(ValueError):
            validate(cfg)
        with self.assertRaises(ValueError):
            Scheme(weight_granularity="per_group")
        with self.assertRaises(ValueError):
            Scheme(activation_scale="pot_mse")

    def test_statistics_survive_relocation_but_not_tampering(self):
        from tests.library_fixtures import toy_model, feeds
        from opennpu_quant.quant.storage import save_statistics, read_statistics

        model = toy_model()
        cfg = CalibrationConfig()
        ort = OrtConfig.cpu()
        stats = calibrate(model, feeds, config=cfg, ort=ort)
        with tempfile.TemporaryDirectory() as temp:
            a, b = Path(temp) / "a", Path(temp) / "b"
            save_statistics(stats, a)
            shutil.copytree(a, b)
            shutil.rmtree(a)
            copy = read_statistics(
                b, model, config=cfg, ort=ort, samples_identity=stats.samples_identity
            )
            self.assertEqual(stats.statistics_identity, copy.statistics_identity)
            gen = b / read_json(b / "complete.json")["generation"]
            p = gen / "histograms.npz"
            p.write_bytes(p.read_bytes() + b"changed")
            with self.assertRaisesRegex(ValueError, "modified"):
                read_statistics(b, model, config=cfg, ort=ort)

    def test_bounded_profile_configuration(self):
        self.assertEqual(1, OrtConfig.cpu().profile_samples)
        self.assertEqual(0, OrtConfig.cpu(profile_samples=0).profile_samples)
        with self.assertRaises(ValueError):
            OrtConfig.cuda(profile_samples=0)
        with self.assertRaises(ValueError):
            OrtConfig.cpu(profile_samples=3)

    def test_historical_imagenet_directory_and_overlap(self):
        from opennpu_quant.data.prepare import prepare

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "calibration_images").mkdir()
            (root / "validation_images").mkdir()
            (root / "calibration_images/img_00000009_label_3.jpg").write_bytes(b"train")
            (root / "validation_images/img_00000000_label_2.jpg").write_bytes(
                b"validation"
            )
            with (
                patch("opennpu_quant.data.prepare.COUNTS", {"imagenet": 1}),
                patch(
                    "opennpu_quant.data.prepare.identifiers", return_value=[["9", "3"]]
                ),
            ):
                prepared = prepare("imagenet", root, root / "manifest.json")
                self.assertTrue(
                    prepared["calibration"][0]["path"].startswith("calibration_images/")
                )
                (root / "validation_images/img_00000000_label_2.jpg").write_bytes(
                    b"train"
                )
                with self.assertRaisesRegex(ValueError, "overlap"):
                    prepare("imagenet", root, root / "manifest.json")

    def test_external_bundle_relocation(self):
        from tests.library_fixtures import toy_model
        from opennpu_quant.graph.model import model_identity

        with tempfile.TemporaryDirectory() as temp:
            a, b = Path(temp) / "a", Path(temp) / "b"
            a.mkdir()
            m = toy_model()
            onnx.save_model(
                m,
                a / "model.onnx",
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                location="weights.bin",
                size_threshold=0,
            )
            expected = model_identity(a / "model.onnx")
            shutil.copytree(a, b)
            shutil.rmtree(a)
            self.assertEqual(expected, model_identity(b / "model.onnx"))

    def test_frozen_framework_exception_does_not_change_graph_or_adaround_tolerance(
        self,
    ):
        from opennpu_quant.export.finalize import framework_compare, compare
        from opennpu_quant.models.spec import ModelSpec

        recipe = ModelSpec.load("mobilenet_v2").recipe
        a = np.asarray([[0.0, 1, 2, 3, 4, 5]], np.float32)
        b = a.copy()
        b[0, 0] = 5e-5
        result = framework_compare({"logits": a}, {"logits": b}, recipe)
        self.assertFalse(result["strict_passed"])
        with self.assertRaises(AssertionError):
            compare({"logits": a}, {"logits": b})
        b[0, 0] = 0.001
        with self.assertRaises(ValueError):
            framework_compare({"logits": a}, {"logits": b}, recipe)
        self.assertEqual(1e-5, AdaroundConfig().atol)

    def test_atomic_download_and_archive_escape(self):
        import hashlib, tarfile, io
        from opennpu_quant.fetch import download, extract_archive

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            source.write_bytes(b"payload")
            with self.assertRaises(ValueError):
                download(source.as_uri(), root / "bad", sha256_expected="0" * 64)
            self.assertFalse((root / "bad").exists())
            download(
                source.as_uri(),
                root / "good",
                sha256_expected=hashlib.sha256(b"payload").hexdigest(),
            )
            with tarfile.open(root / "bad.tar", "w") as archive:
                header = tarfile.TarInfo("../escape")
                header.size = 1
                archive.addfile(header, io.BytesIO(b"x"))
            with self.assertRaises(ValueError):
                extract_archive(root / "bad.tar", root / "out")

    def test_model_lock(self):
        from opennpu_quant._locking import RunLock

        with tempfile.TemporaryDirectory() as temp:
            with RunLock(Path(temp) / "lock"):
                with self.assertRaises(RuntimeError):
                    with RunLock(Path(temp) / "lock"):
                        pass

    def test_local_registered_model_runner_resume_report_and_tamper(self):
        from PIL import Image
        import yaml
        from opennpu_quant.paths import Paths
        from opennpu_quant.runner import run
        from opennpu_quant.models.spec import ModelSpec

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paths = Paths.resolve(home=root)
            directory = paths.models / "toy"
            directory.mkdir(parents=True)
            model = h.make_model(
                h.make_graph(
                    [
                        h.make_node("Flatten", ["images"], ["flat"], name="flatten"),
                        h.make_node(
                            "Gemm",
                            ["flat", "weight", "bias"],
                            ["logits"],
                            name="classifier",
                        ),
                    ],
                    "toy",
                    [h.make_tensor_value_info("images", 1, [1, 3, 4, 4])],
                    [h.make_tensor_value_info("logits", 1, [1, 6])],
                    [
                        n.from_array(
                            np.arange(48 * 6, dtype=np.float32).reshape(48, 6) / 300,
                            "weight",
                        ),
                        n.from_array(np.arange(6, dtype=np.float32) / 10, "bias"),
                    ],
                ),
                opset_imports=[h.make_opsetid("", 18)],
                ir_version=10,
            )
            onnx.save_model(model, directory / "network.onnx")
            recipe = copy.deepcopy(ModelSpec.load("resnet18").recipe)
            recipe["name"] = "toy"
            recipe["preprocess"].update(resize=4, crop=4)
            recipe["input"]["shape"] = [1, 3, 4, 4]
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
                    calibration=[
                        dict(id=i, path=f"{i}.png", target=5) for i in range(2)
                    ],
                    validation=[
                        dict(id=i, path=f"{i}.png", target=5) for i in range(2, 4)
                    ],
                    annotations={},
                    reference_bytes_verified=False,
                ),
            )
            config = load_config()
            config["name"] = "toy_run"
            config["models"] = ["toy"]
            config["percentiles"]["toy"] = dict(pot=99.9, float=99.9)
            config["calibration"]["samples"]["imagenet"] = 2
            config["adaround"].update(
                steps=4, batch_size=2, window_samples=2, window_steps=2, device="cpu"
            )
            config["output"]["save_qdq_models"] = True
            from opennpu_quant.models.adapter import ModelEvaluator
            from unittest.mock import PropertyMock

            details = {"per_class": [{"category_id": 1, "AP": 0.5}]}
            with patch.object(
                ModelEvaluator,
                "metric_details",
                new_callable=PropertyMock,
                return_value=details,
            ):
                result = run(config, paths)
            for condition in config["conditions"]:
                recorded = read_json(
                    paths.runs / "toy_run/toy" / condition / "result.json"
                )
                self.assertEqual(details, recorded["metric_details"])
            self.assertEqual(5, len(result["rows"]))
            self.assertTrue(all(r["complete"] for r in result["rows"]))
            with patch(
                "opennpu_quant.runner.quantize", side_effect=AssertionError("reuse")
            ):
                run(config, paths)
            target = paths.runs / "toy_run/toy/pot_rtn/result.json"
            target.write_text("{}")
            regenerated = run(config, paths)
            self.assertEqual(5, len(regenerated["rows"]))
            self.assertEqual("top1", read_json(target)["metric"])
            # Import a relocated statistics bundle using content, model and feed checks.
            stored = next(
                (paths.cache / "calibration/toy").glob("*/complete.json")
            ).parent
            shutil.copytree(stored, paths.cache / "calibration/toy/reference")
            imported = copy.deepcopy(config)
            imported["name"] = "toy_reference"
            imported["calibration"]["source"] = "reference"
            imported["conditions"] = {
                k: imported["conditions"][k] for k in ("fp32", "pot_rtn")
            }
            with patch(
                "opennpu_quant.runner.calibrate",
                side_effect=AssertionError("must reuse verified statistics"),
            ):
                reused = run(imported, paths)
            self.assertEqual(2, len(reused["rows"]))
            # A failed stage records failure, not a permanently running model.
            failed = copy.deepcopy(config)
            failed["name"] = "forced_failure"
            with patch(
                "opennpu_quant.runner._run_model",
                side_effect=RuntimeError("test failure"),
            ):
                with self.assertRaisesRegex(RuntimeError, "test failure"):
                    run(failed, paths)
            state = read_json(paths.runs / "forced_failure/status.json")
            self.assertEqual("failed", state["models"]["toy"]["state"])
            saved = read_json(paths.runs / "toy_run/effective_config.json")
            config["evaluation"]["limit"] = 1
            with self.assertRaisesRegex(ValueError, "changed"):
                run(config, paths)
            self.assertEqual(
                saved, read_json(paths.runs / "toy_run/effective_config.json")
            )


if __name__ == "__main__":
    unittest.main()
