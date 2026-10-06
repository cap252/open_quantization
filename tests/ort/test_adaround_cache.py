from dataclasses import replace
import unittest
import importlib.util
import numpy as np
from opennpu_quant import OrtConfig, AdaroundConfig, quantize
from opennpu_quant.quant import adaround as impl
from opennpu_quant.quant.adaround_cache import (
    ActivationCacheConfig,
    LayerActivationCache,
)
from tests.library_fixtures import toy_model, feeds
from tests.ort.test_adaround import QCONFIG, group_conv, conv_feeds


class CacheTests(unittest.TestCase):
    def fit(self, model, factory, cache, *, steps=16, window_bytes=512 * 1024**2):
        import torch

        q = quantize(model, factory, config=QCONFIG, ort=OrtConfig.cpu())
        signatures, _ = impl._scan_feeds(factory)
        row = impl._targets(model, q.model, q.audit)[0]
        cfg = AdaroundConfig(
            steps=steps,
            window_steps=2,
            window_samples=2,
            batch_size=2,
            window_bytes=window_bytes,
            device="cpu",
        )
        timings = {}
        with impl._torch_policy(torch, cfg, "cpu"):
            codes, detail = impl._fit_layer(
                row,
                model,
                q.model,
                factory,
                signatures,
                cfg,
                OrtConfig.cpu(),
                "cpu",
                cache_config=cache,
                profiling=timings,
            )
        return codes, detail, timings

    @unittest.skipUnless(
        importlib.util.find_spec("torch"),
        "Optional AdaRound Torch dependency is unavailable",
    )
    def test_cache_preserves_codes_and_errors_for_gemm_and_group_conv(self):
        import torch

        torch.set_num_threads(1)
        for model, factory in ((toy_model(), feeds), (group_conv(), conv_feeds)):
            codes, details, _ = self.fit(model, factory, None)
            for budget in (0, 100, 1024**2):
                with self.subTest(budget=budget, graph=model.graph.name):
                    actual, other, timing = self.fit(
                        model, factory, ActivationCacheConfig(host_bytes=budget)
                    )
                    np.testing.assert_array_equal(codes, actual)
                    for key in (
                        "initial_hard_reconstruction_error",
                        "final_hard_reconstruction_error",
                        "changed_codes",
                        "reconstruction_samples_seen",
                        "error_samples",
                    ):
                        self.assertEqual(details[key], other[key], key)
                    if budget == 1024**2:
                        self.assertGreater(timing["cache"]["hits"], 0)
                        self.assertEqual(len(list(factory())), other["parity_samples"])
                    self.assertLessEqual(timing["cache"]["host_peak_bytes"], budget)

    def test_lru_eviction_and_oversized_pair(self):
        c = LayerActivationCache(ActivationCacheConfig(host_bytes=24))
        pair = (np.ones(2, np.float32), np.zeros(2, np.float32))
        c.put(0, pair)
        self.assertIs(c.get(0), pair)
        c.put(1, pair)
        self.assertIsNone(c.get(0))
        self.assertIs(c.get(1), pair)
        c.put(2, (np.ones(50, np.float32), np.zeros(2, np.float32)))
        self.assertEqual(16, c.bytes)
        self.assertEqual(1, c.metrics["evictions"])

    @unittest.skipUnless(
        importlib.util.find_spec("torch"),
        "Optional AdaRound Torch dependency is unavailable",
    )
    def test_changed_replayed_feed_rejected_even_after_cache_fill(self):
        from unittest.mock import patch

        original = impl._Replay.next
        calls = [0]

        def next_bad(replay):
            calls[0] += 1
            if calls[0] == 3:
                replay.signatures[replay.index] = "changed"
            return original(replay)

        with patch.object(impl._Replay, "next", next_bad):
            with self.assertRaisesRegex(ValueError, "replay changed"):
                self.fit(toy_model(), feeds, ActivationCacheConfig())

    @unittest.skipUnless(
        importlib.util.find_spec("torch"),
        "Optional AdaRound Torch dependency is unavailable",
    )
    def test_no_cross_layer_or_call_cache_and_budget_still_enforced(self):
        _, _, first = self.fit(toy_model(), feeds, ActivationCacheConfig())
        _, _, second = self.fit(toy_model(), feeds, ActivationCacheConfig())
        self.assertEqual(first["cache"], second["cache"])
        with self.assertRaises(MemoryError):
            self.fit(toy_model(), feeds, ActivationCacheConfig(), window_bytes=1)

    @unittest.skipUnless(
        importlib.util.find_spec("torch"),
        "Optional AdaRound Torch dependency is unavailable",
    )
    def test_public_api_frozen_encoding_resume_and_policy_invalidation(self):
        import tempfile
        from unittest.mock import patch
        from opennpu_quant import apply_adaround, ActivationCacheConfig
        from opennpu_quant.quant.parameters import verify_parameters

        model = toy_model()
        q = quantize(model, feeds, config=QCONFIG, ort=OrtConfig.cpu())
        before = q.model.SerializeToString()
        original = model.SerializeToString()
        cfg = AdaroundConfig(
            steps=8, window_steps=2, window_samples=2, batch_size=2, device="cpu"
        )
        baseline = apply_adaround(model, q, feeds, config=cfg, ort=OrtConfig.cpu())
        cache = ActivationCacheConfig(host_bytes=1024)
        with tempfile.TemporaryDirectory() as folder:
            actual = apply_adaround(
                model,
                q,
                feeds,
                config=cfg,
                ort=OrtConfig.cpu(),
                activation_cache=cache,
                checkpoint_dir=folder,
            )
            self.assertEqual(
                baseline.model.SerializeToString(), actual.model.SerializeToString()
            )
            self.assertEqual(
                baseline.audit["adaround"]["frozen_encoding_identity"],
                actual.audit["adaround"]["frozen_encoding_identity"],
            )
            self.assertNotEqual(
                baseline.audit["adaround"]["identity"],
                actual.audit["adaround"]["identity"],
            )
            verify_parameters(actual.model, actual.audit)
            with patch.object(
                impl, "_fit_layer", side_effect=AssertionError("must reuse")
            ):
                reused = apply_adaround(
                    model,
                    q,
                    feeds,
                    config=cfg,
                    ort=OrtConfig.cpu(),
                    activation_cache=cache,
                    checkpoint_dir=folder,
                )
            self.assertEqual(
                actual.model.SerializeToString(), reused.model.SerializeToString()
            )
            with self.assertRaisesRegex(ValueError, "Checkpoint"):
                apply_adaround(
                    model,
                    q,
                    feeds,
                    config=cfg,
                    ort=OrtConfig.cpu(),
                    activation_cache=replace(cache, host_bytes=2048),
                    checkpoint_dir=folder,
                )
        self.assertEqual(before, q.model.SerializeToString())
        self.assertEqual(original, model.SerializeToString())

    @unittest.skipUnless(
        importlib.util.find_spec("torch"),
        "Optional AdaRound Torch dependency is unavailable",
    )
    def test_device_admission_and_allocation_failure_fall_back(self):
        import torch
        from unittest.mock import patch

        pairs = [(np.ones(2, np.float32), np.zeros(2, np.float32))]
        c = LayerActivationCache(
            ActivationCacheConfig(device_bytes=32, device_reserve_bytes=100)
        )
        with patch.object(torch.cuda, "mem_get_info", return_value=(110, 1000)):
            self.assertIs(pairs, c.device_window(pairs, "cuda"))
        with (
            patch.object(torch.cuda, "mem_get_info", return_value=(1000, 1000)),
            patch.object(
                torch, "tensor", side_effect=torch.cuda.OutOfMemoryError("test")
            ),
            patch.object(torch.cuda, "empty_cache"),
        ):
            self.assertIs(pairs, c.device_window(pairs, "cuda"))
        self.assertEqual(2, c.metrics["device_fallback_windows"])

    def test_invalid_budget(self):
        for value in (-1, True, 1.5):
            with self.assertRaises(ValueError):
                ActivationCacheConfig(host_bytes=value)


if __name__ == "__main__":
    unittest.main()
