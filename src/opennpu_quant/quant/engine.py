"""Preserve protected activation edges when inserting Q/DQ.

Shared internal tensors receive Q/DQ only on their external consumer paths.
"""

import copy
from collections import Counter, defaultdict
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from opennpu_quant.ort.numerics import (
    compute_scale_zp,
    get_qmin_qmax_for_qType,
    quantize_data,
    quantize_nparray,
)
from opennpu_quant.ort.numerics import adjust_weight_scale
from opennpu_quant.graph.model import load_model
from opennpu_quant.graph.activations import recognize
from opennpu_quant._io import object_hash, sha256
from .config import scale_policies
from .scales import (
    activation_parameters,
    pot_scale,
    pot_candidates,
    choose_weight_scale,
    normal_scale,
    exponents,
)
from .parameters import parameter_identity, verify_parameters
from opennpu_quant.ort.numerics import compute_data_quant_params

# ------------------------------------------------------------------ operator input-role policy
# 'basic' scope: only these operators select tensors; 'all' scope: every operator below does.
BASIC = {"Conv", "MatMul", "Gemm"}
# Operators whose first input is the only feature activation (other inputs are parameters,
# shapes, indices or axes and keep their original values).
FIRST = {
    # weighted layers, pooling, normalization
    "Conv",
    "Gemm",
    "AveragePool",
    "MaxPool",
    "GlobalAveragePool",
    "GlobalMaxPool",
    "BatchNormalization",
    "InstanceNormalization",
    "LayerNormalization",
    "LpNormalization",
    # activations
    "Relu",
    "Clip",
    "LeakyRelu",
    "PRelu",
    "Sigmoid",
    "HardSigmoid",
    "HardSwish",
    "Tanh",
    "Gelu",
    "Mish",
    "Softplus",
    "Softsign",
    "Elu",
    "Celu",
    "Selu",
    "ThresholdedRelu",
    "Shrink",
    "Softmax",
    "LogSoftmax",
    # shape and indexing operators that move feature data
    "Flatten",
    "Reshape",
    "Resize",
    "Slice",
    "Split",
    "Squeeze",
    "Unsqueeze",
    "Transpose",
    "Expand",
    "Gather",
    "GatherElements",
    "Pad",
    # reductions and element-wise unary operators
    "ReduceMean",
    "ReduceSum",
    "ReduceMax",
    "ReduceMin",
    "ReduceL2",
    "Identity",
    "Cast",
    "Abs",
    "Neg",
    "Exp",
    "Log",
    "Sqrt",
    "Reciprocal",
    "Erf",
}
# Operators for which every input can be a feature activation.
ALL_INPUTS = {
    "Add",
    "Sub",
    "Mul",
    "Div",
    "Pow",
    "Min",
    "Max",
    "MatMul",
    "Concat",
    "Equal",
    "Greater",
    "Less",
    "GreaterOrEqual",
    "LessOrEqual",
}
# Operators whose outputs are never quantized (shapes, indices, booleans). The comparisons are
# also in ALL_INPUTS on purpose: their inputs see the quantized activation, their boolean
# outputs stay untouched.
STRUCTURAL = {
    "Constant",
    "Shape",
    "Size",
    "ConstantOfShape",
    "NonZero",
    "Range",
    "Equal",
    "Greater",
    "GreaterOrEqual",
    "Less",
    "LessOrEqual",
    "And",
    "Or",
    "Not",
}
# Constants stay constant through these operators (used to tell parameters from activations).
CONSTANT_PRESERVING = (
    "Identity",
    "Cast",
    "Reshape",
    "Squeeze",
    "Unsqueeze",
    "Transpose",
    "Neg",
)
QDQ_PREFIX = "__matrix_"


def _feature_input_positions(node):
    """Input positions of `node` that may carry feature activations; unknown operators are an error."""
    if node.op_type in FIRST:
        return [0]
    if node.op_type in ALL_INPUTS:
        return list(range(len(node.input)))
    if node.op_type == "Where":
        return [1, 2]
    if node.op_type in STRUCTURAL:
        return []
    raise ValueError("Explicit input-role policy required: " + node.op_type)


# ------------------------------------------------------------------ quantization plan
def infer_model(source):
    """Load (or copy) an FP32 model and run strict shape inference; QDQ models are rejected."""
    model = load_model(source)
    if any(
        node.op_type in ("QuantizeLinear", "DequantizeLinear")
        for node in model.graph.node
    ):
        raise ValueError("FP32 source required")
    return onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)


def layout(model, scope, protection=None):
    """Plan which tensors and which consumer edges receive Q/DQ for `scope` ('basic' or 'all').

    Returns dict(tensors, edges, rows, protection, outputs):
      tensors     sorted names of the tensors to calibrate and quantize
      edges       {(node index, input index): tensor} consumer edges that read the DQ output
      rows        per-node coverage records (they end up in quantization.json)
      protection  result of graph.activations.recognize(); internal edges never get Q/DQ
      outputs     names of the graph outputs
    """
    if scope not in ("basic", "all"):
        raise ValueError(scope)
    protection = protection or recognize(model)
    if not protection["passed"]:
        unresolved = {
            key: protection[key] for key in ("ambiguous", "conflicts", "nested_graphs")
        }
        raise ValueError("Activation audit: " + str(unresolved))
    graph = model.graph
    typed_values = list(graph.input) + list(graph.output) + list(graph.value_info)
    element_types = {
        value.name: value.type.tensor_type.elem_type for value in typed_values
    }
    constant_tensors = _constant_tensor_names(model)

    feature_positions = {}
    selected_tensors = set()
    rows = []
    for node_index, node in enumerate(graph.node):
        feature_inputs = [
            position
            for position in _feature_input_positions(node)
            if position < len(node.input)
            and node.input[position] not in constant_tensors
            and element_types.get(node.input[position]) == TensorProto.FLOAT
        ]
        feature_positions[node_index] = feature_inputs
        selected = scope == "all" or node.op_type in BASIC
        # A Cast from an integer/boolean tensor produces structural data, not a feature.
        structural_cast = node.op_type == "Cast" and element_types.get(
            node.input[0]
        ) not in (TensorProto.FLOAT, TensorProto.FLOAT16, TensorProto.DOUBLE)
        feature_outputs = [
            name
            for name in node.output
            if element_types.get(name) == TensorProto.FLOAT
            and name not in constant_tensors
            and node.op_type not in STRUCTURAL
            and not structural_cast
        ]
        if selected:
            selected_tensors.update(node.input[position] for position in feature_inputs)
            selected_tensors.update(feature_outputs)
        rows.append(
            dict(
                node=node_index,
                name=node.name,
                op_type=node.op_type,
                selected=selected,
                feature_inputs=feature_inputs,
                parameter_inputs=[
                    position
                    for position in range(len(node.input))
                    if position not in feature_inputs
                ],
                feature_outputs=feature_outputs,
            )
        )

    protected_edges = {
        (node_index, position)
        for region in protection["regions"]
        for node_index, position, _ in region["internal_edges"]
    }
    # Every feature consumer of a selected tensor reads the DQ output, also consumers outside the
    # scope (in 'basic' a Relu after a Conv shares the Conv output boundary). Only the internal
    # edges of protected activation regions keep the raw tensor.
    qdq_edges = {
        (node_index, position): name
        for node_index, node in enumerate(graph.node)
        for position, name in enumerate(node.input)
        if name in selected_tensors
        and position in feature_positions[node_index]
        and (node_index, position) not in protected_edges
    }
    graph_outputs = {value.name for value in graph.output}
    selected_tensors &= set(qdq_edges.values()) | graph_outputs
    for row in rows:
        row["quantized_feature_inputs"] = [
            position
            for position in row["feature_inputs"]
            if (row["node"], position) in qdq_edges
        ]
        row["protected_internal_inputs"] = [
            position
            for position in row["feature_inputs"]
            if (row["node"], position) in protected_edges
        ]
    return dict(
        tensors=sorted(selected_tensors),
        edges=qdq_edges,
        rows=rows,
        protection=protection,
        outputs=graph_outputs,
    )


def _constant_tensor_names(model):
    """Initializers, Constant outputs and what stays constant downstream of them (one forward pass)."""
    constant_tensors = {initializer.name for initializer in model.graph.initializer}
    for node in model.graph.node:
        if node.op_type == "Constant":
            constant_tensors.update(node.output)
        elif node.op_type in CONSTANT_PRESERVING and all(
            name in constant_tensors for name in node.input if name
        ):
            constant_tensors.update(node.output)
    return constant_tensors


# ------------------------------------------------------------------ calibration


def _add_calibration_outputs(model, tensor_names):
    """Expose every tensor to calibrate as an additional graph output."""
    graph = model.graph
    value_infos = {
        value.name: value
        for value in list(graph.input) + list(graph.value_info) + list(graph.output)
    }
    existing_outputs = {value.name for value in graph.output}
    for name in tensor_names:
        if name not in existing_outputs:
            graph.output.append(copy.deepcopy(value_infos[name]))


# ------------------------------------------------------------------ QDQ insertion
def activation_params(tensor_range, symmetric):
    """Signed INT8 per-tensor (scale, zero point) with ORT's own arithmetic."""
    qmin, qmax = get_qmin_qmax_for_qType(TensorProto.INT8, symmetric=symmetric)
    zero_point, scale = compute_scale_zp(
        np.asarray(tensor_range["lowest"], np.float32),
        np.asarray(tensor_range["highest"], np.float32),
        qmin,
        qmax,
        symmetric,
    )
    return np.asarray(scale, np.float32), np.asarray(zero_point, np.int8)


def quantize_model(source, ranges, scheme, protection=None):
    """Return a new QDQ model and its audit record without saving files.

    Original nodes, attributes and their order are kept. Three phases:
      1. activation Q/DQ on every planned tensor, rewiring only the planned consumer edges
      2. INT8 weights and INT32 biases of Conv / Gemm / constant MatMul as DequantizeLinear inputs
      3. reassembly of the node and initializer lists, ONNX check, structural verification
    The emitted bytes are deterministic: names, node order and initializer order are fixed.
    """
    activation_policy, weight_policy = scale_policies(scheme)
    scale_audit = None
    if (activation_policy, weight_policy) != ("float", "float"):
        scale_audit = dict(
            activation_scale_policy=activation_policy,
            weight_scale_policy=weight_policy,
            activations=[],
            weights=[],
            biases=[],
            bias_adjusted_channels=0,
        )
    model = infer_model(source)
    original = copy.deepcopy(model)
    plan = layout(model, scheme["scope"], protection)
    if not set(plan["tensors"]) <= set(ranges):
        raise ValueError("Missing calibrated tensors")
    if any(
        name.startswith(QDQ_PREFIX) for node in model.graph.node for name in node.output
    ):
        raise ValueError("Reserved namespace")
    float_initializers = {
        initializer.name: numpy_helper.to_array(initializer)
        for initializer in model.graph.initializer
    }
    # The only state shared by the phases: new initializers, in the order they must be stored.
    new_initializers = []

    qparams, raw_aliases, qdq_after_node, graph_input_qdq = _insert_activation_qdq(
        model, plan, ranges, scheme, new_initializers, scale_audit
    )
    weight_rows, dq_before_node = _quantize_weights_and_biases(
        model,
        original,
        float_initializers,
        qparams,
        scheme,
        new_initializers,
        scale_audit,
    )
    _reassemble_graph(
        model, graph_input_qdq, dq_before_node, qdq_after_node, new_initializers
    )

    plan["raw_aliases"] = raw_aliases
    onnx.checker.check_model(model, full_check=True)
    audit = verify(original, model, plan, scheme)
    passthrough = sorted(
        {v.name for v in model.graph.input} & {v.name for v in model.graph.output}
    )
    if passthrough:
        audit["preserved_passthrough_outputs"] = passthrough
    audit.update(
        weights=weight_rows,
        scheme=scheme,
        range_identity=object_hash(ranges),
        execution_policy="ORT_DISABLE_ALL; FP32 operators and INT8 QDQ lattice",
    )
    audit["parameter_identity"] = parameter_identity(model)
    audit["parameter_nodes"] = sorted(
        node.name
        for node in model.graph.node
        if node.op_type in ("QuantizeLinear", "DequantizeLinear")
    )
    if scale_audit is not None:
        audit["scale_audit"] = scale_audit
    verify_parameters(model, audit)
    return model, audit


def _add_initializer(new_initializers, name, value):
    new_initializers.append(numpy_helper.from_array(np.asarray(value), name))
    return name


def _dequantize_node(new_initializers, prefix, stored, scale, zero, axis=None):
    """DequantizeLinear of `stored`; appends the scale initializer, then the zero point."""
    scale_name = _add_initializer(new_initializers, prefix + "_scale", scale)
    zero_name = _add_initializer(new_initializers, prefix + "_zero", zero)
    # `axis is None` on purpose: channel axis 0 has to be written.
    attributes = {} if axis is None else dict(axis=axis)
    return helper.make_node(
        "DequantizeLinear",
        [stored, scale_name, zero_name],
        [prefix + "_dq"],
        name=prefix + "_DequantizeLinear",
        **attributes,
    )


def _insert_activation_qdq(
    model, plan, ranges, scheme, new_initializers, scale_audit=None
):
    """Phase 1. Returns (qparams, raw_aliases, qdq_after_node, graph_input_qdq)."""
    nodes = model.graph.node
    # Built before any output is renamed below.
    producer_of = {
        output: node_index
        for node_index, node in enumerate(nodes)
        for output in node.output
    }
    consumer_edges = defaultdict(list)
    for edge, name in plan["edges"].items():
        consumer_edges[name].append(edge)

    qparams = {}  # tensor -> (scale, zero point)
    raw_aliases = {}  # graph output -> name of its unquantized value
    qdq_after_node = defaultdict(
        list
    )  # producer index -> [Q, DQ] placed right after it
    graph_input_qdq = []  # Q/DQ of graph inputs, placed first
    # plan['tensors'] is sorted; its order defines the __matrix_a<n> numbering.
    for number, name in enumerate(plan["tensors"]):
        prefix = f"{QDQ_PREFIX}a{number}"
        scale, zero = activation_params(ranges[name], scheme["activation_symmetric"])
        if scale_audit is not None:
            scale, zero, record = activation_parameters(
                ranges[name],
                scale,
                zero,
                scheme["activation_symmetric"],
                scale_audit["activation_scale_policy"],
            )
            record.update(
                tensor=name,
                q_node=prefix + "_QuantizeLinear",
                dq_node=prefix + "_DequantizeLinear",
            )
            scale_audit["activations"].append(record)
        qparams[name] = (scale, zero)
        raw_name = name
        if name in plan["outputs"] and name in producer_of:
            # A graph output keeps its public name on the DQ output; the producer writes an alias.
            raw_name = prefix + "_original"
            raw_aliases[name] = raw_name
            producer = nodes[producer_of[name]]
            producer.output[list(producer.output).index(name)] = raw_name
        dequantize = _dequantize_node(
            new_initializers, prefix, prefix + "_q", scale, zero
        )
        if name in plan["outputs"] and name in producer_of:
            dequantize.output[0] = name
        quantize_node = helper.make_node(
            "QuantizeLinear",
            [raw_name, prefix + "_scale", prefix + "_zero"],
            [prefix + "_q"],
            name=prefix + "_QuantizeLinear",
        )
        # Rewire consumer by consumer; a shared tensor may feed protected and quantized consumers.
        for node_index, position in consumer_edges[name]:
            nodes[node_index].input[position] = dequantize.output[0]
        if name in producer_of:
            qdq_after_node[producer_of[name]].extend([quantize_node, dequantize])
        else:
            graph_input_qdq.extend([quantize_node, dequantize])
    # Consumers that are not QDQ edges (protected internal edges, parameters) read the raw alias.
    for node_index, node in enumerate(nodes):
        for position, name in enumerate(node.input):
            if name in raw_aliases and (node_index, position) not in plan["edges"]:
                node.input[position] = raw_aliases[name]
    return qparams, raw_aliases, qdq_after_node, graph_input_qdq


def _quantize_weights_and_biases(
    model,
    original,
    float_initializers,
    qparams,
    scheme,
    new_initializers,
    scale_audit=None,
):
    """Phase 2. Returns (weight audit rows, dq_before_node)."""
    weight_rows = []
    dq_before_node = defaultdict(
        list
    )  # node index -> [weight DQ, bias DQ] placed right before it
    for node_index, node in enumerate(model.graph.node):
        if node.op_type not in BASIC or len(node.input) < 2:
            continue
        source_node = original.graph.node[
            node_index
        ]  # input names before phase 1 rewired them
        weight_name = source_node.input[1]
        if weight_name not in float_initializers:
            if node.op_type != "MatMul":
                raise ValueError("Initializer weight required: " + node.name)
            continue  # dynamic MatMul: both operands are activations, handled in phase 1
        weight = float_initializers[weight_name]
        attributes = {
            attribute.name: helper.get_attribute_value(attribute)
            for attribute in node.attribute
        }
        if weight.dtype != np.float32:
            raise ValueError("FP32 weight required")
        has_bias = (
            node.op_type in ("Conv", "Gemm")
            and len(source_node.input) > 2
            and source_node.input[2] in float_initializers
        )

        if scheme["weight_per_channel"]:
            channel_axis = _weight_channel_axis(node, attributes, weight)
        else:
            channel_axis = None
        bias_proto = None
        if has_bias:
            bias_proto = next(
                initializer
                for initializer in original.graph.initializer
                if initializer.name == source_node.input[2]
            )
        if scale_audit is not None:
            if not np.isfinite(weight).all() or not weight.size:
                raise ValueError("Empty/nonfinite FP32 weight")
            if has_bias:
                bias = float_initializers[source_node.input[2]]
                output_axis = _weight_channel_axis(node, attributes, weight)
                if bias.dtype != np.float32 or bias.shape != (
                    weight.shape[output_axis],
                ):
                    raise ValueError(
                        "Unsupported bias broadcasting/layout: " + node.name
                    )
                if not np.isfinite(bias).all():
                    raise ValueError("Nonfinite FP32 bias")
        weight_policy = scheme.get("weight_scale_policy", "float")
        adjusted = False
        if weight_policy == "float":
            codes, scale, zero = _quantize_weight(weight, channel_axis)
            base_scale = np.asarray(scale).copy()
            if has_bias:
                adjusted, codes, scale = _refit_weight_scale_for_int32_bias(
                    weight,
                    codes,
                    scale,
                    zero,
                    channel_axis,
                    qparams[source_node.input[0]][0],
                    weight_name,
                    bias_proto,
                    scheme["weight_per_channel"],
                )
            selection = dict(
                base_scale=base_scale.tolist(),
                selected_scale=base_scale.tolist(),
                final_scale=np.asarray(scale).tolist(),
                exponent=exponents(scale),
                bias_adjusted=np.asarray(scale != base_scale).tolist(),
                candidates=[],
            )
        else:
            codes, scale, zero, selection = _quantize_pot_weight(
                weight,
                channel_axis,
                weight_policy,
                qparams[source_node.input[0]][0] if has_bias else None,
                weight_name,
                bias_proto,
            )
            adjusted = bool(np.any(selection["bias_adjusted"]))
        prefix = f"{QDQ_PREFIX}w{node_index}"
        # Stored order per tensor: codes, scale, zero point.
        codes_name = _add_initializer(new_initializers, prefix + "_codes", codes)
        weight_dq = _dequantize_node(
            new_initializers, prefix, codes_name, scale, zero, channel_axis
        )
        dq_before_node[node_index].append(weight_dq)
        node.input[1] = weight_dq.output[0]
        bias_weight_scale = scale
        if scale_audit is not None:
            selection.update(
                tensor=weight_name,
                consumer=node_index,
                consumer_name=source_node.name,
                policy=weight_policy,
                axis=channel_axis,
                dq_node=weight_dq.name,
                zero_point=np.asarray(zero).tolist(),
            )
            scale_audit["weights"].append(selection)
            scale_audit["bias_adjusted_channels"] += int(
                np.count_nonzero(selection["bias_adjusted"])
            )
        weight_rows.append(
            dict(
                node=node_index,
                tensor=weight_name,
                axis=channel_axis,
                channels=int(np.asarray(scale).size),
                ort_bias_scale_adjusted=bool(adjusted),
            )
        )

        if has_bias:
            if (
                attributes.get("alpha", 1.0) != 1.0
                or attributes.get("beta", 1.0) != 1.0
            ):
                raise ValueError("Non-unit Gemm alpha/beta bias requires support")
            bias_codes, bias_scale = _quantize_bias_int32(
                float_initializers[source_node.input[2]],
                qparams[source_node.input[0]][0],
                bias_weight_scale,
                strict=scale_audit is not None,
            )
            bias_prefix = f"{QDQ_PREFIX}b{node_index}"
            bias_codes_name = _add_initializer(
                new_initializers, bias_prefix + "_codes", bias_codes
            )
            bias_dq = _dequantize_node(
                new_initializers,
                bias_prefix,
                bias_codes_name,
                bias_scale,
                np.zeros(bias_scale.shape, np.int32),
                0 if bias_scale.ndim else None,
            )
            dq_before_node[node_index].append(bias_dq)
            node.input[2] = bias_dq.output[0]
            if scale_audit is not None:
                activation = next(
                    row
                    for row in scale_audit["activations"]
                    if row["tensor"] == source_node.input[0]
                )
                scale_audit["biases"].append(
                    dict(
                        tensor=source_node.input[2],
                        consumer=node_index,
                        dq_node=bias_dq.name,
                        activation_dq_node=activation["dq_node"],
                        weight_dq_node=weight_dq.name,
                        final_scale=bias_scale.tolist(),
                        exponent=exponents(bias_scale),
                        pot_required=scale_audit["activation_scale_policy"] != "float"
                        and weight_policy != "float",
                    )
                )
    return weight_rows, dq_before_node


def _weight_channel_axis(node, attributes, weight):
    """Output-channel axis: 0 for Conv and for Gemm with transB, otherwise the last axis."""
    if node.op_type == "Conv" or (
        node.op_type == "Gemm" and attributes.get("transB", 0)
    ):
        return 0
    return weight.ndim - 1


def _quantize_weight(weight, channel_axis):
    """Symmetric signed INT8 codes with ORT's quantize_data; per channel when an axis is given."""
    if channel_axis is None:
        zero, scale, codes = quantize_data(weight, TensorProto.INT8, symmetric=True)
        return codes, scale, zero
    channel_codes, scales, zeros = [], [], []
    for channel in np.moveaxis(weight, channel_axis, 0):
        zero, scale, codes = quantize_data(channel, TensorProto.INT8, symmetric=True)
        channel_codes.append(codes)
        scales.append(scale)
        zeros.append(zero)
    codes = np.moveaxis(np.stack(channel_codes), 0, channel_axis)
    return codes, np.asarray(scales, np.float32), np.asarray(zeros, np.int8)


def _quantize_pot_weight(weight, channel_axis, policy, input_scale, name, bias_proto):
    channels = (
        [weight] if channel_axis is None else np.moveaxis(weight, channel_axis, 0)
    )
    bases = [
        compute_data_quant_params(channel.ravel(), TensorProto.INT8, symmetric=True)[1]
        for channel in channels
    ]
    base = (
        np.asarray(bases, np.float32)
        if channel_axis is not None
        else np.asarray(bases[0], np.float32)
    )
    initial = [
        pot_candidates(value)[0]
        if policy == "pot_mse"
        else float(pot_scale(value, policy))
        for value in np.asarray(base).reshape(-1)
    ]
    initial = np.asarray(initial, np.float32).reshape(base.shape)
    required = initial.copy()
    if bias_proto is not None:
        _, required = adjust_weight_scale(
            input_scale, required, name, bias_proto, channel_axis is not None
        )

    def encode(values, scale):
        return quantize_nparray(TensorProto.INT8, values, scale, np.asarray(0, np.int8))

    records, values = [], []
    codes = np.empty_like(weight, dtype=np.int8)
    output_channels = (
        [codes] if channel_axis is None else np.moveaxis(codes, channel_axis, 0)
    )
    for i, (channel, output) in enumerate(zip(channels, output_channels)):
        scale, record = choose_weight_scale(
            channel, base.reshape(-1)[i], policy, required.reshape(-1)[i], encode
        )
        output[...] = encode(channel, scale)
        records.append(record)
        values.append(scale)
    scale = np.asarray(values, np.float32).reshape(base.shape)
    result = {
        key: [row[key] for row in records]
        if channel_axis is not None
        else records[0][key]
        for key in (
            "base_scale",
            "bias_required_scale",
            "selected_scale",
            "final_scale",
            "exponent",
            "bias_adjusted",
            "candidates",
        )
    }
    return codes, scale, np.zeros(scale.shape, np.int8), result


def _refit_weight_scale_for_int32_bias(
    weight,
    codes,
    scale,
    zero,
    channel_axis,
    input_scale,
    weight_name,
    bias_proto,
    per_channel,
):
    """Let ORT enlarge a tiny weight scale so that bias / (input scale * weight scale) fits INT32."""
    # ORT's private helper is called without a quantizer instance; it can modify the scale array
    # in place, hence the copy. Re-check this call when the pinned ORT version changes.
    adjusted, new_scale = adjust_weight_scale(
        input_scale, np.asarray(scale).copy(), weight_name, bias_proto, per_channel
    )
    if not adjusted:
        return adjusted, codes, scale
    scale = new_scale
    if channel_axis is None:
        codes = quantize_nparray(TensorProto.INT8, weight, scale, zero)
    else:
        broadcast_shape = [1] * weight.ndim
        broadcast_shape[channel_axis] = scale.size
        codes = quantize_nparray(
            TensorProto.INT8,
            weight,
            scale.reshape(broadcast_shape),
            zero.reshape(broadcast_shape),
        )
    return adjusted, codes, scale


def _quantize_bias_int32(bias, input_scale, weight_scale, *, strict=False):
    """INT32 bias codes with scale = input scale * weight scale (zero point 0)."""
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        bias_scale = np.asarray(input_scale * weight_scale, np.float32)
    if strict:
        normal_scale(bias_scale)
    with np.errstate(over="ignore", invalid="ignore"):
        rounded = np.rint(bias / bias_scale)
    if strict:
        wide = rounded.astype(np.float64)
        if (
            not np.isfinite(wide).all()
            or np.any(wide < -(2.0**31))
            or np.any(wide > 2.0**31 - 1)
        ):
            raise ValueError("INT32 bias overflow")
        return rounded.astype(np.int32), bias_scale
    if not np.isfinite(rounded).all() or np.any(abs(rounded) > np.iinfo(np.int32).max):
        raise ValueError("INT32 bias overflow")
    return rounded.astype(np.int32), bias_scale


def _reassemble_graph(
    model, graph_input_qdq, dq_before_node, qdq_after_node, new_initializers
):
    """Phase 3. Node order: input Q/DQ, then per original node its weight/bias DQ, itself, its output Q/DQ."""
    nodes = list(graph_input_qdq)
    for node_index, node in enumerate(model.graph.node):
        nodes.extend(dq_before_node[node_index])
        nodes.append(node)
        nodes.extend(qdq_after_node[node_index])
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    # Float weights and biases that were replaced by codes are dropped; new initializers go last.
    used = {name for node in nodes for name in node.input} | {
        value.name for value in model.graph.output
    }
    retained = [
        initializer
        for initializer in model.graph.initializer
        if initializer.name in used
    ]
    del model.graph.initializer[:]
    model.graph.initializer.extend(retained + new_initializers)


# ------------------------------------------------------------------ verification
def verify(original, model, plan, scheme):
    """Structural audit of the emitted graph; any violation raises. Returns the audit record."""
    ordinary_nodes = [
        node for node in model.graph.node if not node.name.startswith(QDQ_PREFIX)
    ]
    # 1. Original operators, domains and attributes are untouched and in the same order.
    if len(ordinary_nodes) != len(original.graph.node):
        raise ValueError("Original operators changed")
    for before, after in zip(original.graph.node, ordinary_nodes):
        if (
            before.op_type != after.op_type
            or before.domain != after.domain
            or list(before.attribute) != list(after.attribute)
        ):
            raise ValueError("Operator semantics changed")
    # 2. Internal edges of protected activation regions still read the raw tensor.
    raw_aliases = plan.get("raw_aliases", {})
    for region in plan["protection"]["regions"]:
        for node_index, position, name in region["internal_edges"]:
            if ordinary_nodes[node_index].input[position] != raw_aliases.get(
                name, name
            ):
                raise ValueError("QDQ inside " + region["kind"])
    # 3. Every planned consumer edge reads a DequantizeLinear output.
    dequantized = {
        output
        for node in model.graph.node
        if node.op_type == "DequantizeLinear"
        for output in node.output
    }
    if any(
        ordinary_nodes[node_index].input[position] not in dequantized
        for node_index, position in plan["edges"]
    ):
        raise ValueError("Missing QDQ boundary")
    # 4. Coverage rows: which parameter inputs became INT8 weights / INT32 biases.
    for row in plan["rows"]:
        node = ordinary_nodes[row["node"]]
        row["quantized_weight_inputs"] = [
            position
            for position, name in enumerate(node.input)
            if name.startswith(QDQ_PREFIX + "w") and name.endswith("_dq")
        ]
        row["quantized_bias_inputs"] = [
            position
            for position, name in enumerate(node.input)
            if name.startswith(QDQ_PREFIX + "b") and name.endswith("_dq")
        ]
        row["preserved_parameter_inputs"] = [
            position
            for position in row["parameter_inputs"]
            if position
            not in row["quantized_weight_inputs"] + row["quantized_bias_inputs"]
        ]
    # 5. Scale / zero point / axis of every Q and DQ node.
    parameter_rows = _checked_qdq_parameter_rows(model, scheme)
    return dict(
        passed=True,
        internal_qdq_count=0,
        protected_regions=plan["protection"],
        raw_aliases=plan.get("raw_aliases", {}),
        operator_coverage=plan["rows"],
        quantized_tensors=plan["tensors"],
        operators=dict(Counter(node.op_type for node in model.graph.node)),
        parameters=parameter_rows,
    )


def _checked_qdq_parameter_rows(model, scheme):
    arrays = {
        initializer.name: numpy_helper.to_array(initializer)
        for initializer in model.graph.initializer
    }
    parameter_rows = []
    for node in model.graph.node:
        if node.op_type not in ("QuantizeLinear", "DequantizeLinear"):
            continue
        scale, zero = (arrays[name] for name in node.input[1:3])
        axis = next(
            (attribute.i for attribute in node.attribute if attribute.name == "axis"),
            None,
        )
        if (
            not np.isfinite(scale).all()
            or np.any(scale <= 0)
            or zero.dtype not in (np.int8, np.int32)
        ):
            raise ValueError("Invalid QDQ parameters")
        if node.name.startswith(QDQ_PREFIX + "a"):
            # Activations: per-tensor signed INT8; a symmetric scheme must have zero point 0.
            if (
                scale.ndim
                or zero.ndim
                or zero.dtype != np.int8
                or (scheme["activation_symmetric"] and zero.item() != 0)
            ):
                raise ValueError("Activation parameter mismatch")
        elif np.any(zero):
            raise ValueError("Asymmetric weight/bias")
        if (
            axis is not None
            and node.input[0] in arrays
            and arrays[node.input[0]].shape[axis] != scale.size
        ):
            raise ValueError("Channel axis mismatch")
        parameter_rows.append(
            dict(
                node=node.name,
                axis=axis,
                scale_shape=list(scale.shape),
                zero_min=int(zero.min()),
                zero_max=int(zero.max()),
            )
        )
    return parameter_rows


def audit_runtime(source, executed, quantization):
    """Inspect the serialized ORT graph, including required provider transforms."""
    if not quantization.get("parameter_identity"):
        raise ValueError(
            "Runtime audit requires parameter identity; regenerate the QDQ audit"
        )
    original = onnx.load(source)
    runtime = onnx.load(executed)
    ordinary_nodes = [
        node for node in original.graph.node if not node.name.startswith(QDQ_PREFIX)
    ]
    runtime_nodes_by_name = {node.name: node for node in runtime.graph.node}
    raw_aliases = quantization.get("raw_aliases", {})
    for region in quantization["protected_regions"]["regions"]:
        for node_index, position, name in region["internal_edges"]:
            original_node = ordinary_nodes[node_index]
            runtime_node = (
                runtime_nodes_by_name.get(original_node.name)
                if original_node.name
                else None
            )
            if runtime_node is None:
                matches = [
                    n
                    for n in runtime.graph.node
                    if n.op_type == original_node.op_type
                    and list(n.output) == list(original_node.output)
                ]
                runtime_node = matches[0] if len(matches) == 1 else None
            if runtime_node is None or runtime_node.input[position] != raw_aliases.get(
                name, name
            ):
                raise ValueError(
                    "Runtime changed protected activation boundary: " + region["kind"]
                )
    parameter_check = verify_parameters(runtime, quantization, runtime=True)
    return dict(
        passed=True,
        internal_qdq_count=0,
        parameter_verification=parameter_check,
        sha256=sha256(executed),
        operators=dict(Counter(node.op_type for node in runtime.graph.node)),
        optimization="ORT_DISABLE_ALL",
    )
