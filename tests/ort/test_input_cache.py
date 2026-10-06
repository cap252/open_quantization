from pathlib import Path
from itertools import islice
import tempfile
import unittest
import importlib.util
from unittest.mock import patch
import numpy as np
from opennpu_quant.data.feed_cache import prepare_feed_cache
from opennpu_quant import OrtConfig, quantize, QuantizationConfig, CalibrationConfig
from tests.library_fixtures import toy_model, feeds


class InputCacheTests(unittest.TestCase):
    def cache(self, directory, factory=feeds, **overrides):
        args = dict(
            identity={"transform": "test"},
            verify_source=lambda: None,
            expected_samples=4,
            max_bytes=100000,
        )
        args.update(overrides)
        return prepare_feed_cache(Path(directory) / "cache", factory, **args)

    def test_exact_read_only_replay_and_reuse_without_decode(self):
        with tempfile.TemporaryDirectory() as d:
            cache = self.cache(d)
            for original, cached in zip(feeds(), cache()):
                np.testing.assert_array_equal(original["x"], cached["x"])
                self.assertFalse(cached["x"].flags.writeable)
            reused = self.cache(
                d, lambda: (_ for _ in ()).throw(AssertionError("no decode"))
            )
            self.assertEqual(cache.manifest, reused.manifest)
            self.assertEqual(4, len(list(reused())))

    def test_cache_tamper_is_not_silently_reused(self):
        with tempfile.TemporaryDirectory() as d:
            cache = self.cache(d)
            p = cache.directory / "tensor_0.npy"
            p.chmod(0o644)
            with p.open("r+b") as f:
                f.seek(-1, 2)
                f.write(b"x")
            with self.assertRaises(ValueError):
                list(cache())
            with self.assertRaises(ValueError):
                self.cache(d)

    def test_verified_digest_reuse_detects_replaced_arrays_and_late_mutation(self):
        from opennpu_quant.quant.calibration import feed_digest

        with tempfile.TemporaryDirectory() as d:
            cache = self.cache(d)
            iterator = cache()
            feed = next(iterator)
            expected = feed_digest(next(feeds()))
            self.assertEqual(expected, feed_digest(feed))
            with self.assertRaises(TypeError):
                feed["x"] = np.zeros((1, 4), np.float32)
            with self.assertRaises(ValueError):
                feed["x"].flags.writeable = True
            dict.__setitem__(feed, "x", np.zeros((1, 4), np.float32))
            with self.assertRaises(ValueError):
                feed_digest(feed)
            iterator.close()
            iterator = cache()
            feed = next(iterator)
            path = cache.directory / "tensor_0.npy"
            path.chmod(0o644)
            with path.open("r+b") as stream:
                stream.seek(-1, 2)
                stream.write(b"z")
            with self.assertRaises(ValueError):
                feed_digest(feed)
            with self.assertRaises(ValueError):
                next(iterator)

    def test_corrupt_signature_and_changed_reuse_limits_fail(self):
        from opennpu_quant._io import atomic_json
        import hashlib

        with tempfile.TemporaryDirectory() as d:
            cache = self.cache(d)
            with self.assertRaises(ValueError):
                self.cache(d, expected_samples=3)
            with self.assertRaises(MemoryError):
                self.cache(d, max_bytes=1)
            manifest = cache.manifest
            manifest["feed_digests"][0] = "0" * 64
            manifest["samples_identity"] = hashlib.sha256(
                "".join(manifest["feed_digests"]).encode()
            ).hexdigest()
            atomic_json(cache.directory / "manifest.json", manifest)
            with self.assertRaises(ValueError):
                self.cache(d)

    def test_source_is_checked_on_every_replay(self):
        flag = [False]

        def source():
            if flag[0]:
                raise ValueError("source changed")

        with tempfile.TemporaryDirectory() as d:
            cache = self.cache(d, verify_source=source)
            list(cache())
            flag[0] = True
            with self.assertRaises(ValueError):
                list(cache())

    def test_changed_identity_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            self.cache(d)
            with self.assertRaises(ValueError):
                self.cache(d, identity={"transform": "different"})

    def test_failed_build_leaves_no_committed_payload(self):
        def fail():
            yield next(feeds())
            raise RuntimeError("interrupted")

        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(RuntimeError):
                self.cache(d, fail)
            self.assertFalse((Path(d) / "cache").exists())
            self.assertFalse(list(Path(d).glob("*.building-*")))
            self.assertEqual(4, len(list(self.cache(d)())))

    def test_count_shape_dtype_nonfinite_and_budget_rejected(self):
        factories = [
            lambda: islice(feeds(), 3),
            lambda: iter([{"x": np.zeros((1, 5), np.float32)}, *list(feeds())[1:]]),
            lambda: iter([{"x": np.ones((1, 4), np.float16)}] * 4),
            lambda: iter([{"x": np.full((1, 4), np.nan, np.float32)}] * 4),
        ]
        for factory in factories:
            with tempfile.TemporaryDirectory() as d:
                with self.assertRaises(ValueError):
                    self.cache(d, factory)
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(MemoryError):
                self.cache(d, max_bytes=1)

    def test_raw_and_cached_quantization_identical(self):
        q = QuantizationConfig(
            scope="all",
            activation_symmetric=False,
            weight_granularity="per_channel",
            activation_scale_policy="pot_ceil",
            weight_scale_policy="pot_ceil",
            calibration=CalibrationConfig.percentile(99.99),
        )
        with tempfile.TemporaryDirectory() as d:
            cache = self.cache(d)
            model = toy_model()
            ort = OrtConfig.cpu()
            a = quantize(model, feeds, config=q, ort=ort)
            b = quantize(model, cache, config=q, ort=ort)
            self.assertEqual(a.model.SerializeToString(), b.model.SerializeToString())

    def test_same_cache_writer_lock_excludes_a_second_writer(self):
        from opennpu_quant._locking import RunLock

        with tempfile.TemporaryDirectory() as d:
            with RunLock(Path(d) / "cache.lock"):
                with self.assertRaisesRegex(RuntimeError, "Another process"):
                    self.cache(d)

    def test_no_duck_typed_digest_trust(self):
        from opennpu_quant.quant.calibration import feed_digest

        class Fake(dict):
            def verified_digest(self):
                return "unverified"

        raw = next(feeds())
        self.assertEqual(feed_digest(raw), feed_digest(Fake(raw)))

    def test_boolean_count_and_budget_are_rejected(self):
        for kw in (dict(expected_samples=True), dict(max_bytes=True)):
            with tempfile.TemporaryDirectory() as d:
                with self.assertRaises(ValueError):
                    self.cache(d, **kw)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "Torch not installed")
    def test_adaround_cache_codes_frozen_parameters_and_resume(self):
        from opennpu_quant import apply_adaround, AdaroundConfig, ActivationCacheConfig
        from opennpu_quant.quant import adaround
        from tests.ort.test_adaround import group_conv, conv_feeds
        import torch

        torch.set_num_threads(1)
        ort = OrtConfig.cpu()
        for model, factory, count in (
            (toy_model(), feeds, 4),
            (group_conv(), conv_feeds, 3),
        ):
            for policy in ("float", "pot_ceil"):
                qconfig = QuantizationConfig(
                    scope="all",
                    weight_granularity="per_channel",
                    activation_scale_policy=policy,
                    weight_scale_policy=policy,
                )
                original = model.SerializeToString()
                rtn = quantize(model, factory, config=qconfig, ort=ort)
                config = AdaroundConfig(
                    steps=16,
                    batch_size=2,
                    window_samples=2,
                    window_steps=2,
                    device="cpu",
                )
                ac = ActivationCacheConfig(host_bytes=1024**2)
                baseline = apply_adaround(
                    model, rtn, factory, config=config, ort=ort, activation_cache=ac
                )
                with tempfile.TemporaryDirectory() as d:
                    cache = self.cache(d, factory, expected_samples=count)
                    checkpoint = Path(d) / "adaround"
                    actual = apply_adaround(
                        model,
                        rtn,
                        cache,
                        config=config,
                        ort=ort,
                        activation_cache=ac,
                        checkpoint_dir=checkpoint,
                    )
                    self.assertEqual(
                        baseline.model.SerializeToString(),
                        actual.model.SerializeToString(),
                    )
                    self.assertEqual(
                        baseline.audit["adaround"]["frozen_encoding_identity"],
                        actual.audit["adaround"]["frozen_encoding_identity"],
                    )
                    with patch.object(
                        adaround,
                        "_fit_layer",
                        side_effect=AssertionError("must resume"),
                    ):
                        resumed = apply_adaround(
                            model,
                            rtn,
                            cache,
                            config=config,
                            ort=ort,
                            activation_cache=ac,
                            checkpoint_dir=checkpoint,
                        )
                    self.assertEqual(
                        actual.model.SerializeToString(),
                        resumed.model.SerializeToString(),
                    )
                self.assertEqual(original, model.SerializeToString())

    @unittest.skipUnless(importlib.util.find_spec("torch"), "Torch not installed")
    def test_interrupted_layer_prefix_resumes_with_cached_feeds(self):
        import onnx
        from opennpu_quant import apply_adaround, AdaroundConfig
        from opennpu_quant.quant import adaround
        import torch

        torch.set_num_threads(1)
        model = toy_model()
        model.graph.node.append(
            onnx.helper.make_node(
                "Gemm", ["logits", "tail_weight"], ["final"], name="tail"
            )
        )
        model.graph.initializer.append(
            onnx.numpy_helper.from_array(np.eye(6, dtype=np.float32), "tail_weight")
        )
        model.graph.output[0].name = "final"
        ort = OrtConfig.cpu()
        q = quantize(model, feeds, config=QuantizationConfig(scope="all"), ort=ort)
        config = AdaroundConfig(
            steps=8, window_steps=2, window_samples=2, batch_size=2, device="cpu"
        )
        baseline = apply_adaround(model, q, feeds, config=config, ort=ort)
        fit = adaround._fit_layer
        calls = []

        def interrupt(*args, **kwargs):
            if calls:
                raise RuntimeError("test layer interruption")
            calls.append(True)
            return fit(*args, **kwargs)

        with tempfile.TemporaryDirectory() as folder:
            cache = self.cache(folder)
            checkpoint = Path(folder) / "checkpoints"
            with patch.object(adaround, "_fit_layer", side_effect=interrupt):
                with self.assertRaisesRegex(RuntimeError, "layer interruption"):
                    apply_adaround(
                        model,
                        q,
                        cache,
                        config=config,
                        ort=ort,
                        checkpoint_dir=checkpoint,
                    )
            from opennpu_quant._io import read_json

            self.assertEqual(1, len(read_json(checkpoint / "completed.json")["layers"]))
            resumed = apply_adaround(
                model, q, cache, config=config, ort=ort, checkpoint_dir=checkpoint
            )
            self.assertEqual(
                baseline.model.SerializeToString(), resumed.model.SerializeToString()
            )

    def test_input_cache_is_optional_and_enabled_by_packaged_config(self):
        from copy import deepcopy
        from opennpu_quant.configuration import load_config, validate

        official = load_config("core10")
        self.assertTrue(official["calibration"]["input_cache"]["enabled"])
        uncached = deepcopy(official)
        uncached["calibration"].pop("input_cache")
        validate(uncached)
        for cache in (
            {"enabled": "false"},
            {"max_bytes": 0},
            {"max_bytes": True},
            {"typo": 1},
        ):
            changed = deepcopy(official)
            changed["calibration"]["input_cache"] = cache
            with self.assertRaises(ValueError):
                validate(changed)


class InputCacheCliTests(unittest.TestCase):
    def test_independent_cli_cache_switches(self):
        import json
        import subprocess
        import sys

        result = subprocess.run(
            [
                sys.executable,
                "-B",
                "-m",
                "opennpu_quant",
                "run",
                "--models",
                "resnet18",
                "--no-input-cache",
                "--dry-run",
            ],
            check=True,
            text=True,
            capture_output=True,
        )
        config = json.loads(result.stdout)["config"]
        self.assertFalse(config["calibration"]["input_cache"]["enabled"])
        self.assertTrue(config["activation_cache"]["enabled"])
