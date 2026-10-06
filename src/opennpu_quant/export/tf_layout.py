import numpy as np
from onnx import helper, numpy_helper
from ..graph.partition import extract

VOC_CLASSES = list(range(21))


def tensorflow_final_graph(model, spec):
    """Rename the input, keep NCHW logits and add the graph's own final resize with a static size."""

    graph = model.graph
    old_input = graph.input[0].name
    for node in graph.node:
        for position, name in enumerate(node.input):
            if name == old_input:
                node.input[position] = "images"
    graph.input[0].name = "images"
    # tf2onnx computes in NCHW and transposes back to NHWC at the very end; drop that transpose.
    producers = {name: node for node in graph.node for name in node.output}
    last = producers[graph.output[0].name]
    if last.op_type != "Transpose" or list(last.attribute[0].ints) != [0, 2, 3, 1]:
        raise ValueError("Expected a final NCHW->NHWC transpose in the converted graph")
    coarse_logits = last.input[0]
    resize_node = producers[coarse_logits]
    modes = {
        attribute.name: helper.get_attribute_value(attribute)
        for attribute in resize_node.attribute
    }
    if (
        resize_node.op_type != "Resize"
        or modes.get("coordinate_transformation_mode") != b"align_corners"
    ):
        raise ValueError(
            "Expected the align_corners logits resize of the DeepLab export"
        )
    graph.node.remove(last)
    size = spec["input_size"]
    # ResizeBilinear_3 of the frozen graph: bilinear, align_corners, to the (padded) input size.
    graph.initializer.append(
        numpy_helper.from_array(
            np.array([1, len(VOC_CLASSES), size, size], np.int64), "final_logits_size"
        )
    )
    graph.node.append(
        helper.make_node(
            "Resize",
            [coarse_logits, "", "", "final_logits_size"],
            ["logits"],
            name="final_logits_resize",
            mode="linear",
            coordinate_transformation_mode="align_corners",
        )
    )
    del graph.output[:]
    graph.output.append(
        helper.make_tensor_value_info("logits", 1, [1, len(VOC_CLASSES), size, size])
    )
    return model


def static_logits_resize(model):
    """Give the final logits Resize a constant `sizes` input and drop the shape arithmetic it replaced.

    Resolve shape arithmetic for static logits so the extracted host graph only resizes.
    """

    graph = model.graph
    initializers = {initializer.name for initializer in graph.initializer}
    shapes = {
        value.name: [dim.dim_value for dim in value.type.tensor_type.shape.dim]
        for value in [*graph.value_info, *graph.output]
    }
    for node in graph.node:
        if (
            node.op_type == "Resize"
            and node.output[0] == "logits"
            and node.input[3] not in initializers
        ):
            sizes_name = node.name + "_static_sizes"
            graph.initializer.append(
                numpy_helper.from_array(
                    np.array(shapes["logits"], np.int64), sizes_name
                )
            )
            node.input[2] = ""
            node.input[3] = sizes_name
    return extract(model, ["images"], ["logits"])


def postprocess_inputs(producers, output_name):
    """Walk back from the logits through Resize nodes; the first non-Resize tensor is the network output."""
    name = output_name
    while name in producers and producers[name].op_type == "Resize":
        name = producers[name].input[0]
    if name == output_name:
        raise ValueError("The graph does not end in a logits Resize")
    return [name]
