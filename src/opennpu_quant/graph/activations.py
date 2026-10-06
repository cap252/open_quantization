"""Recognize activations without rewriting their arithmetic.

Ambiguous gates are reported for review. A region's internal_edges identify
consumer edges that must never receive Q/DQ.
"""

from collections import Counter
from dataclasses import dataclass
import math

import numpy as np
import onnx
from onnx import helper, numpy_helper


NATIVE = {
    "Relu": "relu",
    "Clip": "clip",
    "LeakyRelu": "leaky_relu",
    "PRelu": "prelu",
    "Sigmoid": "sigmoid",
    "HardSigmoid": "hard_sigmoid",
    "HardSwish": "hard_swish",
    "Tanh": "tanh",
    "Gelu": "gelu",
    "Mish": "mish",
    "Softplus": "softplus",
    "Softsign": "softsign",
    "Elu": "elu",
    "Celu": "celu",
    "Selu": "selu",
    "ThresholdedRelu": "threshold",
    "Shrink": "shrink",
    "Softmax": "softmax",
    "LogSoftmax": "log_softmax",
}
COMMUTATIVE = {"Add", "Mul", "Max", "Min", "And", "Or"}
BOOLEAN_OUTPUT_OPS = (
    "Equal",
    "Greater",
    "Less",
    "GreaterOrEqual",
    "LessOrEqual",
    "Or",
    "And",
    "Not",
)
# A gate whose sigmoid branch contains pooling and a projection is a squeeze-and-excitation
# block, not one activation; the walk towards the inputs stops after this many producers.
UPSTREAM_WALK_LIMIT = 24


# Pattern mini-language. A pattern is matched against the tensor that a node produces:
#   'X'                      variable: binds a tensor name; every later 'X' must be the same tensor
#   1., .5, ...              scalar constant with exactly this value
#   ('Op', p0, p1, ...)      a node of this operator type whose inputs match p0, p1, ...
#   param('beta')            any finite constant; the same name must have the same value everywhere
#   act('sigmoid', p)        an already recognized activation region of that kind whose input matches p
def param(name):
    return ("parameter", name)


def act(kind, x="X"):
    return ("activation", kind, x)


# Patterns match exact graph operations; variants are not graph rewrites.
# Order matters: when several patterns match the same output, the last one wins.
PATTERNS = [
    ("sigmoid", ("Div", 1.0, ("Add", 1.0, ("Exp", ("Neg", "X"))))),
    ("sigmoid", ("Reciprocal", ("Add", 1.0, ("Exp", ("Neg", "X"))))),
    ("tanh", ("Sub", ("Mul", 2.0, act("sigmoid", ("Mul", 2.0, "X"))), 1.0)),
    (
        "tanh",
        (
            "Div",
            ("Sub", ("Exp", ("Mul", 2.0, "X")), 1.0),
            ("Add", ("Exp", ("Mul", 2.0, "X")), 1.0),
        ),
    ),
    ("silu", ("Mul", "X", act("sigmoid"))),
    ("swish", ("Mul", "X", act("sigmoid", ("Mul", param("beta"), "X")))),
    (
        "gelu_erf",
        ("Mul", 0.5, ("Mul", "X", ("Add", 1.0, ("Erf", ("Div", "X", math.sqrt(2)))))),
    ),
    (
        "gelu_erf",
        ("Mul", ("Mul", 0.5, "X"), ("Add", 1.0, ("Erf", ("Div", "X", math.sqrt(2))))),
    ),
    (
        "gelu_erf",
        (
            "Mul",
            0.5,
            ("Mul", "X", ("Add", 1.0, ("Erf", ("Mul", "X", 1 / math.sqrt(2))))),
        ),
    ),
    (
        "gelu_erf",
        (
            "Mul",
            ("Mul", 0.5, "X"),
            ("Add", 1.0, ("Erf", ("Mul", "X", 1 / math.sqrt(2)))),
        ),
    ),
    ("hard_sigmoid", ("Div", ("Clip", ("Add", "X", 3.0), 0.0, 6.0), 6.0)),
    ("hard_sigmoid", ("Mul", ("Clip", ("Add", "X", 3.0), 0.0, 6.0), 1 / 6)),
    ("hard_sigmoid", ("Clip", ("Add", ("Mul", "X", 1 / 6), 0.5), 0.0, 1.0)),
    ("hard_swish", ("Mul", "X", act("hard_sigmoid"))),
    ("hard_swish", ("Div", ("Mul", "X", ("Clip", ("Add", "X", 3.0), 0.0, 6.0)), 6.0)),
    ("hard_swish", ("Mul", ("Mul", "X", ("Clip", ("Add", "X", 3.0), 0.0, 6.0)), 1 / 6)),
    ("hard_mish", ("Mul", ("Mul", 0.5, "X"), ("Clip", ("Add", "X", 2.0), 0.0, 2.0))),
    ("hard_mish", ("Mul", 0.5, ("Mul", "X", ("Clip", ("Add", "X", 2.0), 0.0, 2.0)))),
    ("softplus", ("Log", ("Add", 1.0, ("Exp", "X")))),
    (
        "softplus",
        (
            "Add",
            ("Max", "X", 0.0),
            ("Log", ("Add", 1.0, ("Exp", ("Neg", ("Abs", "X"))))),
        ),
    ),
    (
        "softplus_beta",
        ("Div", act("softplus", ("Mul", param("beta"), "X")), param("beta")),
    ),
    (
        "softplus_threshold",
        (
            "Where",
            ("Greater", ("Mul", param("beta"), "X"), param("threshold")),
            "X",
            act("softplus_beta"),
        ),
    ),
    (
        "softplus_threshold",
        ("Where", ("Greater", "X", param("threshold")), "X", act("softplus")),
    ),
    ("mish", ("Mul", "X", act("tanh", act("softplus")))),
    ("log_sigmoid", ("Log", act("sigmoid"))),
    ("log_sigmoid", ("Neg", act("softplus", ("Neg", "X")))),
    ("softsign", ("Div", "X", ("Add", 1.0, ("Abs", "X")))),
    ("tanhshrink", ("Sub", "X", act("tanh"))),
    ("relu", ("Max", "X", 0.0)),
    ("relu", ("Where", ("Greater", "X", 0.0), "X", 0.0)),
    ("clip", ("Min", ("Max", "X", param("low")), param("high"))),
    ("clip", ("Max", ("Min", "X", param("high")), param("low"))),
    ("leaky_relu", ("Where", ("Greater", "X", 0.0), "X", ("Mul", param("slope"), "X"))),
    (
        "leaky_relu",
        ("Add", ("Max", "X", 0.0), ("Mul", param("slope"), ("Min", "X", 0.0))),
    ),
    (
        "elu",
        (
            "Where",
            ("Greater", "X", 0.0),
            "X",
            ("Mul", param("alpha"), ("Sub", ("Exp", "X"), 1.0)),
        ),
    ),
    (
        "celu",
        (
            "Add",
            ("Max", "X", 0.0),
            (
                "Min",
                0.0,
                (
                    "Mul",
                    param("alpha"),
                    ("Sub", ("Exp", ("Div", "X", param("alpha"))), 1.0),
                ),
            ),
        ),
    ),
    ("selu", ("Mul", param("scale"), act("elu"))),
    ("threshold", ("Where", ("Greater", "X", param("threshold")), "X", param("value"))),
    (
        "hardshrink",
        (
            "Where",
            (
                "Or",
                ("Greater", "X", param("lambda")),
                ("Less", "X", ("Neg", param("lambda"))),
            ),
            "X",
            0.0,
        ),
    ),
    (
        "softshrink",
        (
            "Add",
            (
                "Where",
                ("Greater", "X", param("lambda")),
                ("Sub", "X", param("lambda")),
                0.0,
            ),
            (
                "Where",
                ("Less", "X", ("Neg", param("lambda"))),
                ("Add", "X", param("lambda")),
                0.0,
            ),
        ),
    ),
]
# tanh-approximated GELU, with x**3 written either as Pow or as two multiplications.
for cube in [("Pow", "X", 3.0), ("Mul", "X", ("Mul", "X", "X"))]:
    tanh = act(
        "tanh", ("Mul", math.sqrt(2 / math.pi), ("Add", "X", ("Mul", 0.044715, cube)))
    )
    PATTERNS += [
        ("gelu_tanh", ("Mul", 0.5, ("Mul", "X", ("Add", 1.0, tanh)))),
        ("gelu_tanh", ("Mul", ("Mul", 0.5, "X"), ("Add", 1.0, tanh))),
    ]

PATTERNS += [
    ("sigmoid", ("Pow", ("Add", 1.0, ("Exp", ("Neg", "X"))), -1.0)),
    (
        "tanh",
        (
            "Div",
            ("Sub", ("Exp", "X"), ("Exp", ("Neg", "X"))),
            ("Add", ("Exp", "X"), ("Exp", ("Neg", "X"))),
        ),
    ),
    (
        "log_sigmoid",
        (
            "Sub",
            ("Min", "X", 0.0),
            ("Log", ("Add", 1.0, ("Exp", ("Neg", ("Abs", "X"))))),
        ),
    ),
    ("leaky_relu_max", ("Max", "X", ("Mul", param("slope"), "X"))),
    (
        "leaky_relu",
        ("Sub", act("relu"), ("Mul", param("slope"), act("relu", ("Neg", "X")))),
    ),
]


def clip_variant(pattern, reverse=False):
    """The same pattern with every Clip written as Min(Max(..)) or, reversed, Max(Min(..))."""
    if not isinstance(pattern, tuple):
        return pattern
    if pattern[0] == "Clip":
        _, x, low, high = pattern
        if reverse:
            return ("Max", ("Min", clip_variant(x, reverse), high), low)
        return ("Min", ("Max", clip_variant(x, reverse), low), high)
    return tuple(clip_variant(part, reverse) for part in pattern)


# Exporters also decompose Clip; add both decompositions of every pattern that contains one.
for kind, pattern in list(PATTERNS):
    for reverse in (False, True):
        variant = clip_variant(pattern, reverse)
        if variant != pattern:
            PATTERNS.append((kind, variant))


@dataclass
class Region:
    kind: str
    inputs: list
    output: str
    nodes: set
    parameters: dict
    evidence: str = "exact operator/data-flow/parameter pattern"


class Recognizer:
    """Finds activation regions of one graph. Node sets are built with the same set operations
    throughout: the order of a region's `internal_edges` follows the iteration order of its
    node set and is part of stored manifests, so node sets are never sorted or rebuilt."""

    # ------------------------------------------------------------------ construction
    def __init__(self, model, gate_provenance=None):
        self.model = model
        self.nodes = list(model.graph.node)
        self.producer = {
            output: index
            for index, node in enumerate(self.nodes)
            for output in node.output
        }
        self.constants = {
            initializer.name: numpy_helper.to_array(initializer)
            for initializer in model.graph.initializer
        }
        typed_values = (
            list(model.graph.input)
            + list(model.graph.value_info)
            + list(model.graph.output)
        )
        self.types = {
            value.name: value.type.tensor_type.elem_type for value in typed_values
        }
        self.shapes = {
            value.name: [
                dim.dim_value if dim.HasField("dim_value") else None
                for dim in value.type.tensor_type.shape.dim
            ]
            for value in typed_values
        }
        for node in self.nodes:
            if node.op_type in BOOLEAN_OUTPUT_OPS:
                self.types.update(
                    {output: onnx.TensorProto.BOOL for output in node.output}
                )
        self.regions = {}  # output tensor -> Region; insertion order feeds the selection
        self.gate_provenance = gate_provenance or {}
        self.ambiguous = []
        self.excluded_gates = []
        self._fold_constants()

    def _fold_constants(self):
        """Make constants visible through Constant nodes and value-preserving wrappers (one pass)."""
        for node in self.nodes:
            if node.op_type == "Constant":
                attrs = self.attrs(node)
                if "value" in attrs:
                    self.constants[node.output[0]] = numpy_helper.to_array(
                        attrs["value"]
                    )
            elif (
                node.op_type in ("Identity", "Cast", "Unsqueeze", "Squeeze", "Reshape")
                and node.input[0] in self.constants
            ):
                # Scalar/parameter identity is sufficient for matching; broadcasting is
                # still validated on the complete original graph by ONNX inference.
                value = self.constants[node.input[0]]
                if node.op_type == "Cast":
                    value = value.astype(
                        helper.tensor_dtype_to_np_dtype(self.attrs(node)["to"])
                    )
                self.constants[node.output[0]] = value
            elif node.op_type == "Neg" and node.input[0] in self.constants:
                self.constants[node.output[0]] = -self.constants[node.input[0]]

    @staticmethod
    def attrs(node):
        return {
            attribute.name: helper.get_attribute_value(attribute)
            for attribute in node.attribute
        }

    # ------------------------------------------------------------------ pattern matching
    def match(self, tensor, pattern, bindings, matched_nodes):
        """True when the producers of `tensor` match `pattern`; extends bindings and matched_nodes."""
        # 1. See through Identity / same-type Cast first. Transactional: work on copies and commit
        #    only on success, so a failed attempt leaves the caller's state untouched.
        producer_index = self.producer.get(tensor)
        if producer_index is not None:
            producer = self.nodes[producer_index]
            if self._is_transparent_wrapper(producer) and tensor not in self.constants:
                trial_bindings, trial_nodes = bindings.copy(), matched_nodes.copy()
                if self.match(producer.input[0], pattern, trial_bindings, trial_nodes):
                    bindings.clear()
                    bindings.update(trial_bindings)
                    matched_nodes.update(trial_nodes)
                    matched_nodes.add(producer_index)
                    return True
        # 2. Variable: bind the tensor name, or require the earlier binding.
        if isinstance(pattern, str):
            if pattern in bindings:
                return bindings[pattern] == tensor
            bindings[pattern] = tensor
            return True
        # 3. Scalar literal: a one-element constant of exactly this value (compared in its dtype).
        if isinstance(pattern, (int, float)):
            value = self.constants.get(tensor)
            return (
                value is not None
                and value.size == 1
                and float(value.item()) == float(np.asarray(pattern, dtype=value.dtype))
            )
        op, *args = pattern
        # 4. Neg(parameter) on a constant that was already folded: match the negated value.
        #    The virtual constant stays registered; no node is added for it.
        if (
            op == "Neg"
            and tensor in self.constants
            and isinstance(args[0], tuple)
            and args[0][0] == "parameter"
        ):
            virtual = "__negative_parameter_" + tensor
            self.constants[virtual] = -self.constants[tensor]
            return self.match(virtual, args[0], bindings, matched_nodes)
        # 5. Named parameter: any finite constant; the same name must carry the same value.
        if op == "parameter":
            value = self.constants.get(tensor)
            if value is None or not np.isfinite(value).all():
                return False
            key = "@" + args[0]
            if key in bindings:
                return np.array_equal(value, self.constants[bindings[key]])
            bindings[key] = tensor
            return True
        # 6. Nested activation: an already recognized single-input region of the requested kind.
        if op == "activation":
            region = self.regions.get(tensor)
            compatible = region is not None and (
                region.kind == args[0]
                or (args[0] == "softplus" and region.kind == "softplus_threshold")
            )
            if not compatible or len(region.inputs) != 1:
                return False
            if not self.match(region.inputs[0], args[1], bindings, matched_nodes):
                return False
            matched_nodes.update(region.nodes)
            return True
        # 7. Operator node.
        return self._match_operator(tensor, op, args, pattern, bindings, matched_nodes)

    def _is_transparent_wrapper(self, node):
        return node.op_type == "Identity" or (
            node.op_type == "Cast"
            and self.attrs(node).get("to") == self.types.get(node.input[0])
        )

    def _match_operator(self, tensor, op, args, pattern, bindings, matched_nodes):
        index = self.producer.get(tensor)
        if index is None:
            return False
        node = self.nodes[index]
        if self._is_transparent_wrapper(node):
            # Reached for constants behind a wrapper (step 1 skips them). Not transactional:
            # the recursion works on the caller's bindings directly.
            if self.match(node.input[0], pattern, bindings, matched_nodes):
                matched_nodes.add(index)
                return True
            return False
        if node.op_type != op:
            return False
        inputs = list(node.input)
        if op == "Clip" and len(inputs) == 1:
            # Opset < 11 keeps the bounds in attributes; expose them as virtual constants.
            attrs = self.attrs(node)
            for bound in ("min", "max"):
                key = f"__clip_attribute_{index}_{bound}"
                self.constants[key] = np.asarray(
                    attrs.get(bound, -np.inf if bound == "min" else np.inf), np.float32
                )
                inputs.append(key)
        if len(inputs) != len(args):
            return False
        orders = [inputs]
        if op in COMMUTATIVE and len(inputs) == 2:
            orders.append(inputs[::-1])
        for order in orders:
            trial_bindings, trial_nodes = bindings.copy(), matched_nodes.copy()
            # all() over a generator on purpose: stop at the first input that does not match.
            if all(
                self.match(name, part, trial_bindings, trial_nodes)
                for name, part in zip(order, args)
            ):
                bindings.clear()
                bindings.update(trial_bindings)
                matched_nodes.update(trial_nodes)
                matched_nodes.add(index)
                return True
        return False

    # ------------------------------------------------------------------ structural recognizers
    def reduction_axis(self, node):
        """Normalized, sorted reduction axes of a keepdims reduction, or None when they are unknown."""
        attrs = self.attrs(node)
        axes = (
            self.constants.get(node.input[1])
            if len(node.input) > 1
            else attrs.get("axes")
        )
        if axes is None or attrs.get("keepdims", 1) != 1:
            return None
        axes = tuple(np.asarray(axes).reshape(-1).tolist())
        shape = self.shapes.get(node.input[0])
        if shape:
            if any(axis < -len(shape) or axis >= len(shape) for axis in axes):
                return None
            axes = tuple(axis % len(shape) for axis in axes)
        if len(set(axes)) != len(axes):
            return None
        return tuple(sorted(axes))

    def slice_halves(self, first_index, second_index):
        """Only contiguous, equal halves of the same statically shaped value."""
        if first_index is None or second_index is None:
            return None
        first, second = self.nodes[first_index], self.nodes[second_index]
        if (
            first.op_type != "Slice"
            or second.op_type != "Slice"
            or first.input[0] != second.input[0]
        ):
            return None
        shape = self.shapes.get(first.input[0])
        if not shape:
            return None
        parts = []
        for node in (first, second):
            attrs = self.attrs(node)
            arguments = []
            for position, key, default in [
                (1, "starts", None),
                (2, "ends", None),
                (3, "axes", [0]),
                (4, "steps", [1]),
            ]:
                value = (
                    self.constants.get(node.input[position])
                    if len(node.input) > position
                    else attrs.get(key, default)
                )
                if value is None or np.asarray(value).size != 1:
                    return None
                arguments.append(int(np.asarray(value).item()))
            start, end, axis, step = arguments
            if not -len(shape) <= axis < len(shape) or step != 1:
                return None
            axis %= len(shape)
            length = shape[axis]
            if length is None or length % 2 or length == 0:
                return None
            start, end, _ = slice(start, end, step).indices(length)
            parts.append((axis, start, end, length))
        axis, _, _, length = parts[0]
        if set(parts) != {
            (axis, 0, length // 2, length),
            (axis, length // 2, length, length),
        }:
            return None
        return first.input[0]

    def softmax(self, index, node):
        """Decomposed softmax: exp(z) / sum(exp(z)), optionally with z = x - max(x)."""
        if node.op_type != "Div":
            return None
        exp_index, sum_index = (self.producer.get(name) for name in node.input)
        if exp_index is None or sum_index is None:
            return None
        exp_node, sum_node = self.nodes[exp_index], self.nodes[sum_index]
        if (
            exp_node.op_type != "Exp"
            or sum_node.op_type != "ReduceSum"
            or sum_node.input[0] != exp_node.output[0]
        ):
            return None
        axes = self.reduction_axis(sum_node)
        if axes is None:
            return None
        source = exp_node.input[0]
        region_nodes = {index, exp_index, sum_index}
        shift_index = self.producer.get(source)
        if shift_index is not None and self.nodes[shift_index].op_type == "Sub":
            shift_node = self.nodes[shift_index]
            max_index = self.producer.get(shift_node.input[1])
            if max_index is not None:
                max_node = self.nodes[max_index]
                if (
                    max_node.op_type == "ReduceMax"
                    and max_node.input[0] == shift_node.input[0]
                    and self.reduction_axis(max_node) == axes
                ):
                    region_nodes.update([shift_index, max_index])
                    source = shift_node.input[0]
                elif max_node.op_type == "ReduceMax":
                    self.ambiguous.append(
                        {
                            "node": index,
                            "output": node.output[0],
                            "reason": "Softmax stabilization reduction axes differ",
                        }
                    )
                    return None
        return Region(
            "softmax", [source], node.output[0], region_nodes, {"axes": list(axes)}
        )

    def gate(self, index, node):
        """GLU family: data * activation(other half). SE gates are excluded, unproven gates reported."""
        if node.op_type != "Mul":
            return None
        variants = {
            "sigmoid": "glu",
            "relu": "reglu",
            "gelu": "geglu",
            "gelu_erf": "geglu",
            "gelu_tanh": "geglu",
            "silu": "swiglu",
            "swish": "swiglu",
        }
        operand_orders = (list(node.input), list(node.input)[::-1])
        # A sigmoid gate fed by pooling + projection is squeeze-and-excitation: never one activation.
        for data_tensor, gate_tensor in operand_orders:
            gate_region = self.regions.get(gate_tensor)
            if gate_region is None or gate_region.kind not in (
                "sigmoid",
                "hard_sigmoid",
            ):
                continue
            if self._is_squeeze_excite(
                self._upstream_op_types(gate_region.inputs, data_tensor)
            ):
                self.excluded_gates.append(
                    {"node": index, "reason": "SE pooling/projection gate"}
                )
                return None
        for data_tensor, gate_tensor in operand_orders:
            if data_tensor in self.constants:
                continue
            gate_region = self.regions.get(gate_tensor)
            if (
                gate_region is None
                or len(gate_region.inputs) != 1
                or gate_region.kind not in variants
            ):
                continue
            gate_source = gate_region.inputs[0]
            if gate_source == data_tensor:
                continue
            data_producer, source_producer = (
                self.producer.get(data_tensor),
                self.producer.get(gate_source),
            )
            provenance = self.gate_provenance.get(node.output[0])
            equal_split = self._equal_split(data_producer, source_producer)
            sliced_source = self.slice_halves(data_producer, source_producer)
            if provenance or equal_split or sliced_source:
                region_nodes = gate_region.nodes | {index}
                region_inputs = [data_tensor, gate_source]
                if equal_split:
                    region_nodes.add(data_producer)
                    region_inputs = [self.nodes[data_producer].input[0]]
                elif sliced_source:
                    region_nodes.update([data_producer, source_producer])
                    region_inputs = [sliced_source]
                if equal_split:
                    evidence = "equal two-way Split"
                elif sliced_source:
                    evidence = "equal contiguous Slice halves"
                else:
                    evidence = str(provenance)
                return Region(
                    variants[gate_region.kind],
                    region_inputs,
                    node.output[0],
                    region_nodes,
                    {},
                    evidence,
                )
            # Known SE branches contain pooling followed by learned projections.
            if self._is_squeeze_excite(
                self._upstream_op_types([gate_source], data_tensor)
            ):
                self.excluded_gates.append(
                    {
                        "node": index,
                        "reason": "SE pooling/projection gate, not a single activation",
                    }
                )
            else:
                self.ambiguous.append(
                    {
                        "node": index,
                        "output": node.output[0],
                        "activation": gate_region.kind,
                        "reason": "Different-input gate needs source provenance",
                    }
                )
            # No return here: the other operand order is still examined.
        return None

    def _upstream_op_types(self, start_tensors, stop_tensor):
        """Operator types found walking from `start_tensors` towards the inputs (bounded walk)."""
        pending = list(start_tensors)
        seen = set()
        op_types = set()
        while pending and len(seen) < UPSTREAM_WALK_LIMIT:
            name = pending.pop()
            producer_index = self.producer.get(name)
            if producer_index is None or producer_index in seen or name == stop_tensor:
                continue
            seen.add(producer_index)
            producer = self.nodes[producer_index]
            op_types.add(producer.op_type)
            pending.extend(producer.input)
        return op_types

    @staticmethod
    def _is_squeeze_excite(op_types):
        return (
            "GlobalAveragePool" in op_types or "ReduceMean" in op_types
        ) and "Conv" in op_types

    def _equal_split(self, data_producer, source_producer):
        """Both gate operands come from one two-way Split with equal halves."""
        equal_split = (
            data_producer is not None
            and data_producer == source_producer
            and self.nodes[data_producer].op_type == "Split"
            and len(self.nodes[data_producer].output) == 2
        )
        if equal_split:
            split_node = self.nodes[data_producer]
            sizes = (
                self.constants.get(split_node.input[1])
                if len(split_node.input) > 1
                else self.attrs(split_node).get("split")
            )
            equal_split = sizes is None or (len(sizes) == 2 and sizes[0] == sizes[1])
            shape = self.shapes.get(split_node.input[0])
            axis = self.attrs(split_node).get("axis", 0)
            if shape and -len(shape) <= axis < len(shape) and shape[axis] is not None:
                equal_split = equal_split and shape[axis] % 2 == 0
        return equal_split

    # ------------------------------------------------------------------ recognition pass
    def run(self):
        """Recognize regions node by node, keep the largest non-overlapping ones and report."""
        nested_graphs = []
        nested_audits = []
        self._audit_functions(nested_graphs, nested_audits)
        # The order of the steps matters: later steps build on regions registered by earlier ones,
        # and a later registration for the same output replaces the earlier one.
        for index, node in enumerate(self.nodes):
            self._audit_subgraphs(index, node, nested_graphs, nested_audits)
            self._register_native(index, node)
            self._register_patterns(index, node)
            self._register_softmax_family(index, node)
            self._register_inverse_wrappers(index, node)
            gate_region = self.gate(index, node)
            if gate_region:
                self.regions[gate_region.output] = gate_region

        selected, occupied, conflicts = self._select_regions()
        rows = [
            self._region_row(region)
            for region in sorted(selected, key=lambda region: min(region.nodes))
        ]
        # Only unresolved gates that survived inside no larger recognized activation.
        ambiguous = [
            item
            for item in self.ambiguous
            if not any(
                item["node"] in region.nodes and len(region.nodes) > 1
                for region in selected
            )
        ]
        ambiguous += [
            {
                "node": index,
                "output": node.output[0],
                "reason": "Unrecognized Erf-based activation candidate",
            }
            for index, node in enumerate(self.nodes)
            if node.op_type == "Erf" and index not in occupied
        ]
        return {
            "regions": rows,
            "counts": dict(Counter(row["kind"] for row in rows)),
            "excluded_gates": self.excluded_gates,
            "ambiguous": ambiguous,
            "conflicts": conflicts,
            "nested_graphs": nested_graphs,
            "nested_audits": nested_audits,
            "nested_policy": "Recursively inspected; nested execution requires an explicit scoped QDQ policy before running",
            "passed": not (ambiguous or conflicts or nested_graphs),
        }

    def _audit_functions(self, nested_graphs, nested_audits):
        """Function bodies are inspected recursively; their presence alone fails the audit."""
        for function in self.model.functions:
            function_name = function.domain + "::" + function.name
            nested_graphs.append(
                {"function": function_name, "nodes": len(function.node)}
            )
            graph = helper.make_graph(
                list(function.node),
                function.name,
                [helper.make_empty_tensor_value_info(name) for name in function.input],
                [helper.make_empty_tensor_value_info(name) for name in function.output],
            )
            body = helper.make_model(
                graph,
                opset_imports=list(function.opset_import)
                or list(self.model.opset_import),
            )
            nested_audits.append(
                {
                    "function": function_name,
                    "audit": Recognizer(body, self.gate_provenance).run(),
                }
            )

    def _audit_subgraphs(self, index, node, nested_graphs, nested_audits):
        for attribute in node.attribute:
            if attribute.type in (
                onnx.AttributeProto.GRAPH,
                onnx.AttributeProto.GRAPHS,
            ):
                nested_graphs.append({"node": index, "attribute": attribute.name})
                graphs = (
                    [attribute.g]
                    if attribute.type == onnx.AttributeProto.GRAPH
                    else attribute.graphs
                )
                for graph in graphs:
                    body = helper.make_model(
                        graph, opset_imports=list(self.model.opset_import)
                    )
                    nested_audits.append(
                        {
                            "node": index,
                            "attribute": attribute.name,
                            "graph": graph.name,
                            "audit": Recognizer(body, self.gate_provenance).run(),
                        }
                    )

    def _register_native(self, index, node):
        """A native ONNX activation operator is a one-node region; record its parameters."""
        if node.op_type not in NATIVE:
            return
        attrs = self.attrs(node)
        if node.op_type == "HardSigmoid":
            attrs = {"alpha": attrs.get("alpha", 0.2), "beta": attrs.get("beta", 0.5)}
        if node.op_type in ("Softmax", "LogSoftmax"):
            opset = next(
                o.version
                for o in self.model.opset_import
                if o.domain in ("", "ai.onnx")
            )
            attrs.update(
                axis=attrs.get("axis", -1 if opset >= 13 else 1),
                opset=opset,
                axis_semantics="single axis"
                if opset >= 13
                else "flatten dimensions from axis",
            )
        if node.op_type == "Clip":
            for position, key in [(1, "min"), (2, "max")]:
                if (
                    len(node.input) > position
                    and node.input[position] in self.constants
                ):
                    value = self.constants[node.input[position]]
                    if value.size == 1:
                        attrs[key] = float(value.item())
            if attrs.get("min") == 0.0 and attrs.get("max") == 6.0:
                attrs["semantic_alias"] = "ReLU6"
        parameters = {
            key: value
            for key, value in attrs.items()
            if isinstance(value, (int, float, str))
        }
        self.regions[node.output[0]] = Region(
            NATIVE[node.op_type],
            [node.input[0]],
            node.output[0],
            {index},
            parameters,
            "native ONNX activation",
        )

    def _register_patterns(self, index, node):
        """Try every pattern on the node output; the last matching pattern defines the region."""
        for kind, pattern in PATTERNS:
            bindings = {}
            matched_nodes = set()
            if (
                self.match(node.output[0], pattern, bindings, matched_nodes)
                and "X" in bindings
            ):
                parameters = {
                    key[1:]: self.constants[name].tolist()
                    for key, name in bindings.items()
                    if key.startswith("@")
                }
                kind = self._specialize_kind(kind, parameters, matched_nodes)
                if kind is None:
                    continue
                self.regions[node.output[0]] = Region(
                    kind, [bindings["X"]], node.output[0], matched_nodes, parameters
                )

    def _specialize_kind(self, kind, parameters, matched_nodes):
        """Refine a matched kind from its parameters; None rejects the match."""
        if kind == "hard_swish":
            for inner in self.regions.values():
                if inner.kind == "hard_sigmoid" and inner.nodes <= matched_nodes:
                    alpha, beta = (
                        inner.parameters.get("alpha", 1 / 6),
                        inner.parameters.get("beta", 0.5),
                    )
                    parameters.update(alpha=alpha, beta=beta)
                    if alpha == 0.5 and beta == 1.0:
                        kind = "hard_mish"
                    elif not math.isclose(alpha, 1 / 6, rel_tol=1e-7) or beta != 0.5:
                        kind = "hard_swish_parametric"
        if kind == "clip" and np.any(
            np.asarray(parameters["low"]) > np.asarray(parameters["high"])
        ):
            return None
        if kind == "leaky_relu_max":
            slope = np.asarray(parameters["slope"])
            if np.any(slope < 0) or np.any(slope > 1):
                return None
            kind = "leaky_relu"
        if kind == "leaky_relu" and np.asarray(parameters.get("slope", 0)).size > 1:
            kind = "prelu"
        if (
            kind.startswith("softplus")
            and float(np.asarray(parameters.get("beta", 1))) <= 0
        ):
            return None
        if kind == "swish" and np.asarray(parameters["beta"]).size == 1:
            beta = float(np.asarray(parameters["beta"]))
            if beta == float(np.float32(1.702)):
                kind = "quick_gelu"
            elif beta == 1.0:
                kind = "silu"
        return kind

    def _register_softmax_family(self, index, node):
        """Decomposed softmax, Log(softmax), stable log-softmax and softmin."""
        softmax_region = self.softmax(index, node)
        if softmax_region:
            self.regions[softmax_region.output] = softmax_region
        if (
            node.op_type == "Log"
            and node.input[0] in self.regions
            and self.regions[node.input[0]].kind == "softmax"
        ):
            inner = self.regions[node.input[0]]
            self.regions[node.output[0]] = Region(
                "log_softmax",
                inner.inputs,
                node.output[0],
                inner.nodes | {index},
                inner.parameters,
            )
        # Stable LogSoftmax: z - log(sum(exp(z))); optional z=x-max(x).
        if node.op_type == "Sub":
            log_index = self.producer.get(node.input[1])
            log_node = self.nodes[log_index] if log_index is not None else None
            sum_index = (
                self.producer.get(log_node.input[0])
                if log_node is not None and log_node.op_type == "Log"
                else None
            )
            sum_node = self.nodes[sum_index] if sum_index is not None else None
            exp_index = (
                self.producer.get(sum_node.input[0])
                if sum_node is not None and sum_node.op_type == "ReduceSum"
                else None
            )
            exp_node = self.nodes[exp_index] if exp_index is not None else None
            if (
                exp_node is not None
                and exp_node.op_type == "Exp"
                and exp_node.input[0] == node.input[0]
                and self.reduction_axis(sum_node) is not None
            ):
                source = node.input[0]
                region_nodes = {index, log_index, sum_index, exp_index}
                shift_index = self.producer.get(source)
                if shift_index is not None and self.nodes[shift_index].op_type == "Sub":
                    shift_node = self.nodes[shift_index]
                    max_index = self.producer.get(shift_node.input[1])
                    max_node = self.nodes[max_index] if max_index is not None else None
                    if (
                        max_node is not None
                        and max_node.op_type == "ReduceMax"
                        and max_node.input[0] == shift_node.input[0]
                        and self.reduction_axis(max_node)
                        == self.reduction_axis(sum_node)
                    ):
                        region_nodes.update([shift_index, max_index])
                        source = shift_node.input[0]
                self.regions[node.output[0]] = Region(
                    "log_softmax",
                    [source],
                    node.output[0],
                    region_nodes,
                    {"axes": list(self.reduction_axis(sum_node))},
                )
        region = self.regions.get(node.output[0])
        if region and region.kind == "softmax":
            negation_index = self.producer.get(region.inputs[0])
            if (
                negation_index is not None
                and self.nodes[negation_index].op_type == "Neg"
            ):
                self.regions[region.output] = Region(
                    "softmin",
                    [self.nodes[negation_index].input[0]],
                    region.output,
                    region.nodes | {negation_index},
                    region.parameters,
                )

    def _register_inverse_wrappers(self, index, node):
        """Transpose/Reshape/Cast pairs that cancel out around an activation belong to its region."""
        if node.op_type == "Transpose":
            region = self.regions.get(node.input[0])
            inner_index = (
                self.producer.get(region.inputs[0])
                if region and region.kind in ("softmax", "log_softmax")
                else None
            )
            inner_node = self.nodes[inner_index] if inner_index is not None else None
            if inner_node is not None and inner_node.op_type == "Transpose":
                inner_perm, outer_perm = (
                    self.attrs(inner_node).get("perm"),
                    self.attrs(node).get("perm"),
                )
                if (
                    inner_perm is not None
                    and outer_perm is not None
                    and list(np.asarray(inner_perm)[outer_perm])
                    == list(range(len(inner_perm)))
                ):
                    self.regions[node.output[0]] = Region(
                        region.kind,
                        [inner_node.input[0]],
                        node.output[0],
                        region.nodes | {inner_index, index},
                        region.parameters,
                        "inverse transpose wrappers around activation",
                    )
        if node.op_type in ("Reshape", "Cast"):
            region = self.regions.get(node.input[0])
            wrapper_index = (
                self.producer.get(region.inputs[0])
                if region and len(region.inputs) == 1
                else None
            )
            wrapper = self.nodes[wrapper_index] if wrapper_index is not None else None
            if wrapper is not None and wrapper.op_type == node.op_type:
                before, after = wrapper.input[0], node.output[0]
                if node.op_type == "Reshape":
                    shape_before, shape_after = (
                        self.shapes.get(before),
                        self.shapes.get(after),
                    )
                    inverse = (
                        shape_before is not None
                        and shape_before == shape_after
                        and None not in shape_before
                        and all(
                            reshape.input[1] in self.constants
                            for reshape in (wrapper, node)
                        )
                    )
                else:
                    # ONNX element types 1, 10, 11: FLOAT, FLOAT16, DOUBLE.
                    inverse = (
                        self.types.get(before) == self.types.get(after)
                        and self.types.get(before) in (1, 10, 11)
                        and self.types.get(wrapper.output[0]) in (1, 10, 11)
                    )
                if inverse:
                    self.regions[after] = Region(
                        region.kind,
                        [before],
                        after,
                        region.nodes | {wrapper_index, index},
                        region.parameters,
                        "verified inverse "
                        + node.op_type
                        + " wrappers; original operations retained",
                    )

    def _select_regions(self):
        """Largest regions first; a region inside a selected one is dropped, a partial overlap is a conflict."""
        regions = sorted(
            self.regions.values(), key=lambda region: len(region.nodes), reverse=True
        )
        selected = []
        occupied = set()
        conflicts = []
        for region in regions:
            if region.nodes <= occupied:
                continue
            if region.nodes & occupied:
                conflicts.append({"kind": region.kind, "output": region.output})
                continue
            selected.append(region)
            occupied.update(region.nodes)
        return selected, occupied, conflicts

    def _region_row(self, region):
        internal = {
            output
            for index in region.nodes
            for output in self.nodes[index].output
            if output != region.output
        }
        # Iterates the node set as it is; see the class docstring before changing this.
        edges = [
            (index, position, name)
            for index in region.nodes
            for position, name in enumerate(self.nodes[index].input)
            if name in internal
        ]
        return {
            "kind": region.kind,
            "inputs": region.inputs,
            "output": region.output,
            "nodes": sorted(region.nodes),
            "internal_tensors": sorted(internal),
            "internal_edges": edges,
            "parameters": region.parameters,
            "evidence": region.evidence,
        }


def recognize(model, gate_provenance=None):
    return Recognizer(model, gate_provenance).run()
