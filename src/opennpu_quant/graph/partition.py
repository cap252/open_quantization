import copy
import onnx
from onnx import helper as h


def free_variables(graph):
    defined = {x.name for x in graph.input} | {x.name for x in graph.initializer}
    defined |= {x for n in graph.node for x in n.output if x}
    used = {x.name for x in graph.output}
    for node in graph.node:
        used.update(dependencies(node))
    return used - defined - {""}


def dependencies(node):
    result = set(node.input) - {""}
    for a in node.attribute:
        if a.type == onnx.AttributeProto.GRAPH:
            result.update(free_variables(a.g))
        elif a.type == onnx.AttributeProto.GRAPHS:
            for graph in a.graphs:
                result.update(free_variables(graph))
    return result


def extract(model, input_names, output_names):
    """Cut only main-graph tensor edges; reject missing inputs instead of leaking CNN into host."""
    model = onnx.shape_inference.infer_shapes(model)
    graph = model.graph
    info = {v.name: v for v in [*graph.value_info, *graph.input, *graph.output]}
    tensors = {t.name: t for t in graph.initializer}
    producers = {v: i for i, n in enumerate(graph.node) for v in n.output}
    boundary = set(input_names)
    needed = set()
    constants = set()
    leaves = set()

    def visit(name):
        if not name:
            return
        if name in boundary:
            leaves.add(name)
            return
        if name in tensors:
            constants.add(name)
            return
        if name not in producers:
            raise ValueError("Unbound graph input/capture: " + name)
        i = producers[name]
        if i in needed:
            return
        needed.add(i)
        for value in dependencies(graph.node[i]):
            visit(value)

    for name in output_names:
        visit(name)
    selected = [n for i, n in enumerate(graph.node) if i in needed]
    selected_inputs = [name for name in input_names if name in leaves]
    used = set(selected_inputs) | set(output_names)
    for n in selected:
        used.update(dependencies(n))
        used.update(n.output)
    result = copy.deepcopy(model)
    new = h.make_graph(
        selected,
        graph.name + "_partition",
        [info[n] for n in selected_inputs],
        [info[n] for n in output_names],
        [t for t in graph.initializer if t.name in constants],
        value_info=[
            v
            for v in graph.value_info
            if v.name in used and v.name not in selected_inputs + output_names
        ],
    )
    result.graph.CopyFrom(new)
    onnx.checker.check_model(result)
    return result


def rename_input(model, before, after):
    result = copy.deepcopy(model)
    if before == after:
        return result
    if any(v.name == after for v in result.graph.input):
        raise ValueError("Input rename conflict")
    for n in result.graph.node:
        for i, value in enumerate(n.input):
            if value == before:
                n.input[i] = after
    for v in [*result.graph.input, *result.graph.value_info]:
        if v.name == before:
            v.name = after
    onnx.checker.check_model(result)
    return result
