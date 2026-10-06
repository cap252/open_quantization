import numpy as np
import onnx
from onnx import helper as h, numpy_helper as n


def toy_model():
    weights = np.asarray(
        [
            [0.1, 0.2, -0.1, 0.05, 0.03, 0.3],
            [0.3, -0.2, 0.1, 0.1, -0.1, 0.2],
            [-0.2, 0.1, 0.2, 0.3, 0.1, -0.3],
            [0.1, 0.2, 0.4, -0.1, 0.2, 0.05],
        ],
        np.float32,
    )
    bias = np.arange(6, dtype=np.float32) / 10
    graph = h.make_graph(
        [
            h.make_node("Gemm", ["x", "w", "b"], ["hidden"], name="projection"),
            h.make_node("Sigmoid", ["hidden"], ["gate"], name="gate"),
            h.make_node("Mul", ["hidden", "gate"], ["logits"], name="activation"),
        ],
        "toy",
        [h.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 4])],
        [h.make_tensor_value_info("logits", onnx.TensorProto.FLOAT, [1, 6])],
        [n.from_array(weights, "w"), n.from_array(bias, "b")],
    )
    return h.make_model(graph, opset_imports=[h.make_opsetid("", 18)], ir_version=10)


def feeds():
    for i in range(4):
        yield {"x": np.asarray([[i / 4, -0.5, 0.3, 0.1]], np.float32)}


def samples():
    from opennpu_quant import EvaluationSample

    for i, feed in enumerate(feeds()):
        yield EvaluationSample(str(i), feed, 5)


def evaluator():
    from opennpu_quant import TopKClassification

    return TopKClassification()


def quantization_matrix():
    from opennpu_quant import CalibrationConfig, QuantizationConfig

    return tuple(
        QuantizationConfig(
            scope=scope,
            activation_symmetric=symmetric,
            weight_granularity=granularity,
            calibration=calibration,
        )
        for scope in ("basic", "all")
        for symmetric in (True, False)
        for granularity in ("per_tensor", "per_channel")
        for calibration in (
            CalibrationConfig(),
            CalibrationConfig.percentile(99.9),
            CalibrationConfig.percentile(99.99),
            CalibrationConfig.percentile(99.999),
            CalibrationConfig.entropy(),
        )
    )
