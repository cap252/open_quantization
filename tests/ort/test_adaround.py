from dataclasses import replace
from pathlib import Path
import copy
import tempfile
import unittest
import importlib.util
from unittest.mock import patch
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as n
from opennpu_quant import (
    AdaroundConfig,
    QuantizationConfig,
    CalibrationConfig,
    OrtConfig,
    quantize,
    calibrate,
    apply_adaround,
)
from opennpu_quant.quant.adaround import frozen_identity, rounding_state, rounding_codes
from opennpu_quant.quant.parameters import verify_parameters
from tests.library_fixtures import toy_model, feeds
from tests.ort.test_pot_quantization import shared_graph, shared_feeds

SMALL = AdaroundConfig(steps=4, batch_size=2, window_samples=2, window_steps=2)
QCONFIG = QuantizationConfig(
    scope="all",
    weight_granularity="per_channel",
    activation_scale_policy="pot_ceil",
    weight_scale_policy="pot_ceil",
)


def group_conv():
    w = np.arange(36, dtype=np.float32).reshape(4, 1, 3, 3) / 33 - 0.4
    b = np.asarray([0.02, -0.1, 0.2, 0], np.float32)
    graph = h.make_graph(
        [
            h.make_node(
                "Conv",
                ["x", "w", "b"],
                ["y"],
                name="depthwise",
                group=2,
                pads=[1, 0, 2, 1],
                strides=[2, 1],
                dilations=[1, 2],
            )
        ],
        "group",
        [h.make_tensor_value_info("x", 1, [1, 2, 7, 8])],
        [h.make_tensor_value_info("y", 1, [1, 4, 4, 5])],
        [n.from_array(w, "w"), n.from_array(b, "b")],
    )
    return h.make_model(graph, opset_imports=[h.make_opsetid("", 17)], ir_version=10)


def conv_feeds():
    for i in range(3):
        yield {
            "x": np.random.default_rng(i).normal(size=(1, 2, 7, 8)).astype(np.float32)
        }


@unittest.skipUnless(
    importlib.util.find_spec("torch"),
    "Optional AdaRound Torch dependency is unavailable",
)
class AdaroundTests(unittest.TestCase):
    def test_configuration_roundtrip_and_default_name(self):
        q = replace(QCONFIG, adaround=SMALL)
        self.assertEqual(q, QuantizationConfig.from_dict(q.to_dict()))
        self.assertIn("__adaround_", q.name)
        self.assertNotIn("adaround", QCONFIG.name)
        self.assertIsNone(QuantizationConfig.from_dict({"scope": "all"}).adaround)
        for options in (
            {"steps": 0},
            {"device": "magic"},
            {"learning_rate": float("nan")},
            {"warmup_fraction": 1},
            {"batch_size": True},
        ):
            with self.assertRaises(ValueError):
                AdaroundConfig(**options)

    def test_only_codes_change_and_public_inputs_untouched(self):
        model = toy_model()
        original = model.SerializeToString()
        rtn = quantize(model, feeds, config=QCONFIG, ort=OrtConfig.cpu())
        before = rtn.model.SerializeToString()
        audit = copy.deepcopy(rtn.audit)
        result = apply_adaround(model, rtn, feeds, config=SMALL, ort=OrtConfig.cpu())
        names = {row["encoded_weight"] for row in result.audit["adaround"]["layers"]}
        self.assertEqual(
            frozen_identity(rtn.model, names), frozen_identity(result.model, names)
        )
        self.assertEqual(original, model.SerializeToString())
        self.assertEqual(before, rtn.model.SerializeToString())
        self.assertEqual(audit, rtn.audit)
        self.assertEqual(rtn.range_identity, result.range_identity)
        verify_parameters(
            onnx.load_from_string(result.model.SerializeToString()), result.audit
        )
        self.assertEqual(0, result.audit["internal_qdq_count"])
        self.assertEqual(
            ["alpha"], result.audit["adaround"]["layers"][0]["gradient_parameters"]
        )
        from opennpu_quant.quant.engine import audit_runtime
        from opennpu_quant.ort.session import create_session

        with tempfile.TemporaryDirectory() as folder:
            a, b = Path(folder) / "a.onnx", Path(folder) / "b.onnx"
            onnx.save(result.model, a)
            session = create_session(a, OrtConfig.cpu(), optimized=b)
            del session
            self.assertTrue(
                audit_runtime(a, b, result.audit)["parameter_verification"]["passed"]
            )

    def test_rounding_integer_saturation_zero_and_gradients(self):
        import torch

        weights = np.array(
            [-1000, -128, -127.4, -1, 0, 0.2, 0.5, 1, 126.9, 127, 1000], np.float32
        )
        alpha, floor, scale, active = rounding_state(
            weights, np.float32(1), device="cpu"
        )
        soft = rounding_codes(alpha, floor, active, hard=False)
        soft.sum().backward()
        self.assertIsNotNone(alpha.grad)
        self.assertIsNone(floor.grad)
        self.assertIsNone(scale.grad)
        self.assertTrue(torch.all(alpha.grad[~active] == 0))
        hard = rounding_codes(alpha, floor, active, hard=True).detach().numpy()
        self.assertTrue(np.all(hard >= -128))
        self.assertTrue(np.all(hard <= 127))
        np.testing.assert_array_equal(hard[[1, 3, 4, 7, 9]], weights[[1, 3, 4, 7, 9]])
        alpha, fl, sc, ac = rounding_state(
            np.zeros(3, np.float32), np.float32(1), device="cpu"
        )
        self.assertFalse(ac.any())
        self.assertTrue((rounding_codes(alpha, fl, ac, hard=True) == 0).all())

    def test_group_conv_stride_dilation_asymmetric_padding(self):
        q = quantize(group_conv(), conv_feeds, config=QCONFIG, ort=OrtConfig.cpu())
        result = apply_adaround(
            group_conv(), q, conv_feeds, config=SMALL, ort=OrtConfig.cpu()
        )
        row = result.audit["adaround"]["layers"][0]
        self.assertEqual(0, row["axis"])
        self.assertGreater(row["parity_samples"], 0)

    def test_gemm_axes_constant_and_dynamic_matmul(self):
        q = quantize(shared_graph(), shared_feeds, config=QCONFIG, ort=OrtConfig.cpu())
        result = apply_adaround(
            shared_graph(), q, shared_feeds, config=SMALL, ort=OrtConfig.cpu()
        )
        self.assertEqual(
            [1, 0, 1], [r["axis"] for r in result.audit["adaround"]["layers"]]
        )
        self.assertEqual(1, len(result.audit["adaround"]["dynamic_matmul"]))
        verify_parameters(result.model, result.audit)

    def test_gemm_transposed_input_batch_axis(self):
        model = toy_model()
        model.graph.node[0].attribute.append(h.make_attribute("transA", 1))
        model.graph.input[0].CopyFrom(h.make_tensor_value_info("x", 1, [4, 1]))

        def transposed():
            for item in feeds():
                yield {"x": item["x"].T.copy()}

        q = quantize(model, transposed, config=QCONFIG, ort=OrtConfig.cpu())
        result = apply_adaround(model, q, transposed, config=SMALL, ort=OrtConfig.cpu())
        self.assertEqual(1, len(result.audit["adaround"]["layers"]))

    def test_factory_required_and_replay_tamper_detected(self):
        model = toy_model()
        stats = calibrate(model, feeds, config=CalibrationConfig(), ort=OrtConfig.cpu())
        with self.assertRaisesRegex(ValueError, "reconstruction_feeds"):
            quantize(
                model,
                stats,
                config=replace(QCONFIG, adaround=SMALL),
                ort=OrtConfig.cpu(),
            )
        count = [0]

        def changing():
            count[0] += 1
            for row in feeds():
                yield {"x": row["x"] + np.float32(count[0])}

        q = quantize(model, stats, config=QCONFIG, ort=OrtConfig.cpu())
        with self.assertRaisesRegex(ValueError, "replay changed"):
            apply_adaround(model, q, changing, config=SMALL, ort=OrtConfig.cpu())

    def test_cached_codes_must_preserve_integer_points(self):
        from opennpu_quant.quant.adaround import validate_hard_codes

        weights = np.asarray([0.5, 0.6, -1000], np.float32)
        validate_hard_codes(weights, np.float32(0.5), np.asarray([1, 2, -128], np.int8))
        for codes in ([2, 2, -128], [1, 3, -128], [1, 2, -127]):
            with self.assertRaisesRegex(ValueError, "floor/ceil"):
                validate_hard_codes(
                    weights, np.float32(0.5), np.asarray(codes, np.int8)
                )

    def test_checkpoint_resume_and_tamper(self):
        model = toy_model()
        q = quantize(model, feeds, config=QCONFIG, ort=OrtConfig.cpu())
        with tempfile.TemporaryDirectory() as folder:
            a = apply_adaround(
                model,
                q,
                feeds,
                config=SMALL,
                ort=OrtConfig.cpu(),
                checkpoint_dir=folder,
            )
            with patch(
                "opennpu_quant.quant.adaround._fit_layer",
                side_effect=AssertionError("must reuse"),
            ):
                b = apply_adaround(
                    model,
                    q,
                    feeds,
                    config=SMALL,
                    ort=OrtConfig.cpu(),
                    checkpoint_dir=folder,
                )
            self.assertEqual(a.model.SerializeToString(), b.model.SerializeToString())
            from opennpu_quant._io import read_json

            path = (
                Path(folder) / read_json(Path(folder) / "completed.json")["codes_file"]
            )
            path.write_bytes(path.read_bytes() + b"modified")
            with self.assertRaisesRegex(ValueError, "checkpoint was modified"):
                apply_adaround(
                    model,
                    q,
                    feeds,
                    config=SMALL,
                    ort=OrtConfig.cpu(),
                    checkpoint_dir=folder,
                )

    def test_nonfinite_and_window_budget(self):
        q = quantize(toy_model(), feeds, config=QCONFIG, ort=OrtConfig.cpu())

        def bad():
            yield {"x": np.full((1, 4), np.nan, np.float32)}

        with self.assertRaisesRegex(ValueError, "nonfinite"):
            apply_adaround(toy_model(), q, bad, config=SMALL, ort=OrtConfig.cpu())
        with self.assertRaises(MemoryError):
            apply_adaround(
                toy_model(),
                q,
                feeds,
                config=replace(SMALL, window_bytes=1),
                ort=OrtConfig.cpu(),
            )

    def test_encoded_shared_weight_rejected(self):
        model = toy_model()
        q = quantize(model, feeds, config=QCONFIG, ort=OrtConfig.cpu())
        shared = onnx.load_from_string(q.model.SerializeToString())
        shared.graph.node.append(
            h.make_node("Identity", ["__matrix_w0_dq"], ["shared"], name="extra")
        )
        q = replace(q, model=shared)
        with self.assertRaisesRegex(ValueError, "Shared encoded"):
            apply_adaround(model, q, feeds, config=SMALL, ort=OrtConfig.cpu())

    def test_quantize_feed_api_equals_explicit(self):
        model = toy_model()
        rtn = quantize(model, feeds, config=QCONFIG, ort=OrtConfig.cpu())
        a = apply_adaround(model, rtn, feeds, config=SMALL, ort=OrtConfig.cpu())
        b = quantize(
            model, feeds, config=replace(QCONFIG, adaround=SMALL), ort=OrtConfig.cpu()
        )
        self.assertEqual(a.model.SerializeToString(), b.model.SerializeToString())


@unittest.skipUnless(
    importlib.util.find_spec("torch"),
    "Optional AdaRound Torch dependency is unavailable",
)
class AdaroundRestartTests(unittest.TestCase):
    def test_only_completed_prefix_is_reused_after_interrupt(self):
        model = toy_model()
        # Append a second independent constant projection after the protected activation.
        model.graph.initializer.append(
            n.from_array(np.eye(6, dtype=np.float32) * 0.3, "w2")
        )
        model.graph.node.append(
            h.make_node("MatMul", ["logits", "w2"], ["final"], name="second")
        )
        model.graph.output[0].CopyFrom(h.make_tensor_value_info("final", 1, [1, 6]))
        rtn = quantize(model, feeds, config=QCONFIG, ort=OrtConfig.cpu())
        from opennpu_quant.quant import adaround as implementation

        original_fit = implementation._fit_layer
        attempts = []

        def interrupted(row, *args, **kwargs):
            attempts.append(row["index"])
            if len(attempts) == 2:
                raise KeyboardInterrupt("simulated interruption")
            return original_fit(row, *args, **kwargs)

        with tempfile.TemporaryDirectory() as folder:
            with patch.object(implementation, "_fit_layer", side_effect=interrupted):
                with self.assertRaises(KeyboardInterrupt):
                    apply_adaround(
                        model,
                        rtn,
                        feeds,
                        config=SMALL,
                        ort=OrtConfig.cpu(),
                        checkpoint_dir=folder,
                    )
            resumed_nodes = []

            def observed(row, *args, **kwargs):
                resumed_nodes.append(row["index"])
                return original_fit(row, *args, **kwargs)

            with patch.object(implementation, "_fit_layer", side_effect=observed):
                resumed = apply_adaround(
                    model,
                    rtn,
                    feeds,
                    config=SMALL,
                    ort=OrtConfig.cpu(),
                    checkpoint_dir=folder,
                )
            uninterrupted = apply_adaround(
                model, rtn, feeds, config=SMALL, ort=OrtConfig.cpu()
            )
            self.assertEqual([3], resumed_nodes)
            self.assertEqual(
                uninterrupted.model.SerializeToString(),
                resumed.model.SerializeToString(),
            )


if __name__ == "__main__":
    unittest.main()
