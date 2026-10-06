from dataclasses import replace
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as n
from opennpu_quant import *
from tests.library_fixtures import quantization_matrix
from opennpu_quant.ort.session import create_session
from opennpu_quant.quant.config import ACTIVATION_SCALE_POLICIES, WEIGHT_SCALE_POLICIES
from opennpu_quant.quant.engine import _quantize_bias_int32, audit_runtime
from opennpu_quant.quant.parameters import verify_parameters
from opennpu_quant.quant.scales import is_pot
from tests.library_fixtures import toy_model, feeds


def shared_graph():
    weights = np.asarray(
        [[0.1, 0.02, -0.15], [0.7, -0.2, 0.3], [-0.1, 0.5, 0.9], [0.4, 0.3, -0.2]],
        np.float32,
    )
    nodes = [
        h.make_node("Gemm", ["x", "w", "b0"], ["y0"], name="normal"),
        h.make_node("Gemm", ["z", "w", "b1"], ["y1"], name="transposed", transB=1),
        h.make_node("MatMul", ["x", "w"], ["y2"], name="constant"),
        h.make_node("MatMul", ["y0", "dynamic"], ["y3"], name="dynamic"),
    ]
    graph = h.make_graph(
        nodes,
        "shared",
        [
            h.make_tensor_value_info("x", 1, [1, 4]),
            h.make_tensor_value_info("z", 1, [1, 3]),
            h.make_tensor_value_info("dynamic", 1, [3, 2]),
        ],
        [
            h.make_tensor_value_info(name, 1, [1, width])
            for name, width in [("y0", 3), ("y1", 4), ("y2", 3), ("y3", 2)]
        ],
        [
            n.from_array(weights, "w"),
            n.from_array(np.zeros(3, np.float32), "b0"),
            n.from_array(np.full(4, 1e8, np.float32), "b1"),
        ],
    )
    return h.make_model(graph, opset_imports=[h.make_opsetid("", 18)], ir_version=10)


def shared_feeds():
    for value in (0.2, 0.7):
        yield {
            "x": np.full((1, 4), value, np.float32),
            "z": np.full((1, 3), value, np.float32),
            "dynamic": np.full((3, 2), value, np.float32),
        }


class PotQuantizationTests(unittest.TestCase):
    def test_all_480_combinations_and_40_golden_outputs(self):
        baseline = json.loads(
            (Path(__file__).parents[1] / "fixtures/pot/golden_40.json").read_text()
        )["schemes"]
        model = toy_model()
        before = model.SerializeToString()
        ort = OrtConfig.cpu()
        stats = calibrate(model, feeds, config=CalibrationConfig(), ort=ort)
        count = 0
        with contextlib.redirect_stdout(io.StringIO()):
            for old in quantization_matrix():
                for activation in ACTIVATION_SCALE_POLICIES:
                    for weight in WEIGHT_SCALE_POLICIES:
                        config = replace(
                            old,
                            activation_scale_policy=activation,
                            weight_scale_policy=weight,
                        )
                        with self.subTest(scheme=config.name, scope=config.scope):
                            result = quantize(model, stats, config=config, ort=ort)
                            onnx.checker.check_model(result.model, full_check=True)
                            values = create_session(result.model, ort).run(
                                None, next(feeds())
                            )
                            self.assertTrue(np.isfinite(values[0]).all())
                            self.assertEqual(0, result.audit["internal_qdq_count"])
                            verify_parameters(
                                onnx.load_from_string(result.model.SerializeToString()),
                                result.audit,
                            )
                            if activation == weight == "float":
                                expected = baseline[old.scope + "/" + old.name]
                                self.assertEqual(
                                    expected["model_sha256"],
                                    hashlib.sha256(
                                        result.model.SerializeToString()
                                    ).hexdigest(),
                                )
                                self.assertEqual(
                                    expected["output_sha256"],
                                    [
                                        hashlib.sha256(v.tobytes()).hexdigest()
                                        for v in values
                                    ],
                                )
                            else:
                                self.assertIn("scale_audit", result.audit)
                            count += 1
        self.assertEqual(480, count)
        self.assertEqual(before, model.SerializeToString())

    def test_shared_weights_different_axes_bias_and_dynamic_matmul(self):
        config = QuantizationConfig(
            scope="all",
            activation_symmetric=False,
            weight_granularity="per_channel",
            activation_scale_policy="pot_ceil",
            weight_scale_policy="pot_mse",
        )
        result = quantize(
            shared_graph(), shared_feeds, config=config, ort=OrtConfig.cpu()
        )
        rows = result.audit["scale_audit"]["weights"]
        self.assertEqual([1, 0, 1], [row["axis"] for row in rows])
        self.assertEqual([3, 4, 3], [len(row["final_scale"]) for row in rows])
        self.assertGreater(result.audit["scale_audit"]["bias_adjusted_channels"], 0)
        self.assertIn(
            "dynamic",
            [row["tensor"] for row in result.audit["scale_audit"]["activations"]],
        )
        self.assertTrue(
            all(
                np.isfinite(v).all()
                for v in create_session(result.model, OrtConfig.cpu()).run(
                    None, next(shared_feeds())
                )
            )
        )

    def test_conv_bias_scale_enlargement_and_mixed_policy(self):
        weight = np.full((2, 1, 1, 1), 2.0**-16, np.float32)
        graph = h.make_graph(
            [h.make_node("Conv", ["x", "w", "b"], ["y"], name="conv")],
            "bias",
            [h.make_tensor_value_info("x", 1, [1, 1, 2, 2])],
            [h.make_tensor_value_info("y", 1, [1, 2, 2, 2])],
            [
                n.from_array(weight, "w"),
                n.from_array(np.array([0, 100], np.float32), "b"),
            ],
        )
        model = h.make_model(
            graph, opset_imports=[h.make_opsetid("", 18)], ir_version=10
        )

        def small():
            yield {"x": np.full((1, 1, 2, 2), 2.0**-16, np.float32)}

        for activation in ("float", "pot_nearest"):
            config = QuantizationConfig(
                weight_granularity="per_channel",
                activation_scale_policy=activation,
                weight_scale_policy="pot_mse",
            )
            result = quantize(model, small, config=config, ort=OrtConfig.cpu())
            audit = result.audit["scale_audit"]
            self.assertEqual(1, audit["bias_adjusted_channels"])
            self.assertEqual([False, True], audit["weights"][0]["bias_adjusted"])
            self.assertEqual(activation != "float", audit["biases"][0]["pot_required"])
            if activation != "float":
                self.assertTrue(is_pot(audit["biases"][0]["final_scale"]))
            self.assertTrue(
                np.isfinite(
                    create_session(result.model, OrtConfig.cpu()).run(
                        None, next(small())
                    )[0]
                ).all()
            )

    def test_int32_and_fp32_product_limits_are_checked_before_cast(self):
        one = np.asarray(1.0, np.float32)
        with self.assertRaisesRegex(ValueError, "INT32"):
            _quantize_bias_int32(np.array([2.0**31], np.float32), one, one, strict=True)
        result, _ = _quantize_bias_int32(
            np.array([-(2.0**31)], np.float32), one, one, strict=True
        )
        self.assertEqual(np.iinfo(np.int32).min, result[0])
        for exponent in (-100, 100):
            with self.assertRaisesRegex(ValueError, "FP32 normal"):
                _quantize_bias_int32(
                    np.array([0], np.float32),
                    np.float32(2.0**exponent),
                    np.float32(2.0**exponent),
                    strict=True,
                )

    def test_parameter_tamper_and_runtime_clones(self):
        q = quantize(
            toy_model(),
            feeds,
            config=QuantizationConfig(
                activation_scale_policy="pot_nearest", weight_scale_policy="pot_mse"
            ),
            ort=OrtConfig.cpu(),
        )
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "model.onnx"
            executed = Path(folder) / "runtime.onnx"
            onnx.save(q.model, source)
            session = create_session(source, OrtConfig.cpu(), optimized=executed)
            del session
            self.assertTrue(
                audit_runtime(source, executed, q.audit)["parameter_verification"][
                    "passed"
                ]
            )
            changed = onnx.load(executed)
            initializer = next(
                v for v in changed.graph.initializer if v.name.endswith("_scale")
            )
            initializer.CopyFrom(
                n.from_array(
                    n.to_array(initializer) * np.float32(1.0001), initializer.name
                )
            )
            with self.assertRaisesRegex(ValueError, "identity"):
                verify_parameters(changed, q.audit, runtime=True)
        changed = onnx.load_from_string(q.model.SerializeToString())
        initializer = next(
            v
            for v in changed.graph.initializer
            if v.name.endswith("_codes") and v.data_type == onnx.TensorProto.INT8
        )
        array = n.to_array(initializer).copy()
        array.flat[0] += 1
        initializer.CopyFrom(n.from_array(array, initializer.name))
        with self.assertRaisesRegex(ValueError, "identity"):
            verify_parameters(changed, q.audit)

    def test_zero_weights_and_unsupported_bias_broadcast(self):
        model = toy_model()
        for initializer in model.graph.initializer:
            initializer.CopyFrom(
                n.from_array(np.zeros_like(n.to_array(initializer)), initializer.name)
            )
        config = QuantizationConfig(
            weight_granularity="per_channel",
            activation_scale_policy="pot_ceil",
            weight_scale_policy="pot_mse",
        )
        result = quantize(model, feeds, config=config, ort=OrtConfig.cpu())
        self.assertEqual(
            [1.0] * 6, result.audit["scale_audit"]["weights"][0]["base_scale"]
        )
        bias = next(v for v in model.graph.initializer if v.name == "b")
        bias.CopyFrom(n.from_array(np.float32(1), "b"))
        with self.assertRaisesRegex(ValueError, "bias broadcasting"):
            quantize(model, feeds, config=config, ort=OrtConfig.cpu())

    def test_activation_fanout_preserves_internal_raw_edge(self):
        model = toy_model()
        model.graph.node.append(
            h.make_node("Neg", ["gate"], ["external"], name="fanout")
        )
        model.graph.output.append(h.make_tensor_value_info("external", 1, [1, 6]))
        config = QuantizationConfig(
            scope="all",
            activation_scale_policy="pot_nearest",
            weight_scale_policy="pot_mse",
        )
        result = quantize(model, feeds, config=config, ort=OrtConfig.cpu())
        activation = next(
            node for node in result.model.graph.node if node.name == "activation"
        )
        fanout = next(node for node in result.model.graph.node if node.name == "fanout")
        self.assertEqual("gate", activation.input[1])
        self.assertNotEqual("gate", fanout.input[0])
        self.assertEqual(0, result.audit["internal_qdq_count"])
        self.assertTrue(
            all(
                np.isfinite(v).all()
                for v in create_session(result.model, OrtConfig.cpu()).run(
                    None, next(feeds())
                )
            )
        )
