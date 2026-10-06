import hashlib
import numpy as np
from onnx import numpy_helper
from opennpu_quant._io import object_hash
from opennpu_quant.graph.model import load_model
from .scales import normal_scale, is_pot


def parameter_identity(model, expected_nodes=None):
    arrays = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}

    def identity(value):
        return dict(
            dtype=str(value.dtype),
            shape=list(value.shape),
            sha256=hashlib.sha256(value.tobytes()).hexdigest(),
        )

    rows = []
    qdq = [
        node
        for node in model.graph.node
        if node.op_type in ("QuantizeLinear", "DequantizeLinear")
    ]
    if expected_nodes is not None:
        expected = {node.name: node for node in qdq if node.name in expected_nodes}
        if set(expected) != set(expected_nodes):
            raise ValueError("ORT removed required QDQ nodes")
        for node in qdq:
            if node.name in expected:
                continue
            # ORT may clone a DQ for multiple consumers even with optimizations disabled.
            matches = [
                other
                for other in expected.values()
                if node.op_type == other.op_type == "DequantizeLinear"
                and node.input[0] == other.input[0]
                and all(
                    np.array_equal(arrays[a], arrays[b])
                    and arrays[a].dtype == arrays[b].dtype
                    for a, b in zip(node.input[1:3], other.input[1:3])
                )
                and _axis(node, arrays) == _axis(other, arrays)
                and _block_size(node) == _block_size(other)
            ]
            if not matches:
                raise ValueError("Unexpected runtime QDQ node: " + node.name)
    for node in model.graph.node:
        if node.op_type not in ("QuantizeLinear", "DequantizeLinear"):
            continue
        if expected_nodes is not None and node.name not in expected_nodes:
            continue
        rows.append(
            dict(
                node=node.name,
                op=node.op_type,
                axis=_axis(node, arrays),
                scale=identity(arrays[node.input[1]]),
                zero=identity(arrays[node.input[2]]),
                codes=identity(arrays[node.input[0]])
                if node.input[0] in arrays
                else None,
            )
        )
        if _block_size(node):
            rows[-1]["block_size"] = _block_size(node)
    return object_hash(sorted(rows, key=lambda row: (row["node"], row["op"])))


def _block_size(node):
    return next((a.i for a in node.attribute if a.name == "block_size"), 0)


def _axis(node, arrays):
    # Axis is ignored for scalar scales; ORT materializes the default axis=1.
    return (
        None
        if arrays[node.input[1]].ndim == 0
        else next((a.i for a in node.attribute if a.name == "axis"), 1)
    )


def verify_parameters(source, audit, *, runtime=False):
    model = load_model(source)
    if audit.get("parameter_identity") != parameter_identity(
        model, audit.get("parameter_nodes") if runtime else None
    ):
        raise ValueError("QDQ parameter identity changed")
    details = audit.get("scale_audit")
    if details is None:
        return dict(passed=True, parameter_identity=audit["parameter_identity"])
    arrays = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}
    nodes = {node.name: node for node in model.graph.node}

    def parameters(name):
        node = nodes[name]
        return arrays[node.input[1]], arrays[node.input[2]]

    for row in details["activations"]:
        scale, zero = parameters(row["dq_node"])
        qscale, qzero = parameters(row["q_node"])
        if nodes[row["dq_node"]].input[0] != nodes[row["q_node"]].output[0]:
            raise ValueError("Activation Q/DQ connection changed")
        if not np.array_equal(scale, qscale) or not np.array_equal(zero, qzero):
            raise ValueError("Activation Q/DQ parameters differ")
        if (
            scale.shape != ()
            or zero.dtype != np.int8
            or float(scale) != row["final_scale"]
            or int(zero) != row["zero_point"]
        ):
            raise ValueError("Activation scale record differs from graph")
        if row["policy"] != "float" and not is_pot(scale):
            raise ValueError("Activation scale is not exactly PoT")
    for row in details["weights"]:
        scale, zero = parameters(row["dq_node"])
        expected = np.asarray(row["final_scale"], np.float32)
        node = nodes[row["dq_node"]]
        axis = _axis(node, arrays)
        if (
            not np.array_equal(scale, expected)
            or axis != row["axis"]
            or np.any(zero)
            or zero.dtype != np.int8
        ):
            raise ValueError("Weight scale/axis record differs from graph")
        codes = arrays[node.input[0]]
        if codes.dtype != np.int8 or (
            axis is not None and codes.shape[axis] != scale.size
        ):
            raise ValueError("Weight code or channel shape mismatch")
        if row["policy"] != "float" and not is_pot(scale):
            raise ValueError("Weight scale is not exactly PoT")
    for row in details["biases"]:
        scale, zero = parameters(row["dq_node"])
        act, _ = parameters(row["activation_dq_node"])
        weight, _ = parameters(row["weight_dq_node"])
        if "weight_group_shape" in row:
            weight = weight.reshape(row["weight_group_shape"]).max(axis=1)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            expected = np.asarray(act * weight, np.float32)
        normal_scale(scale)
        if (
            not np.array_equal(scale, expected)
            or np.any(zero)
            or zero.dtype != np.int32
        ):
            raise ValueError(
                "Bias scale must equal final activation scale times weight scale"
            )
        if arrays[nodes[row["dq_node"]].input[0]].dtype != np.int32:
            raise ValueError("Bias codes must be INT32")
        if row["pot_required"] and not is_pot(scale):
            raise ValueError("Bias scale is not exactly PoT")
    return dict(
        passed=True,
        parameter_identity=audit["parameter_identity"],
        bias_adjusted_channels=details["bias_adjusted_channels"],
    )
