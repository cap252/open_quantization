from pathlib import Path
import numpy as np
from ..fetch import extract_archive


def export_model(recipe, weight, feeds, target, scratch):
    import tensorflow as tf
    import tf2onnx, onnx

    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_intra_op_parallelism_threads(
        2 if recipe["export"]["family"] == "tf_ssd" else 1
    )
    tf.config.threading.set_inter_op_parallelism_threads(1)
    extracted = Path(scratch) / "source"
    extract_archive(weight, extracted)
    if recipe["export"]["family"] == "tf_ssd":
        found = list(extracted.rglob("saved_model.pb"))
        if len(found) != 1:
            raise ValueError("Expected one SSD SavedModel")
        saved_model = tf.saved_model.load(str(found[0].parent))
        original = saved_model.signatures["serving_default"]
        _, kw = original.structured_input_signature
        if len(kw) != 1:
            raise ValueError("Expected one SSD input")
        input_name = next(iter(kw))
        names = sorted(original.structured_outputs)
        if kw[input_name].dtype != tf.uint8:
            raise ValueError("SSD input must be uint8")
        references = [
            {
                name: v.numpy()
                for name, v in original(
                    **{input_name: tf.convert_to_tensor(feed["images"])}
                ).items()
            }
            for feed in feeds
        ]
        signature = [tf.TensorSpec(recipe["input"]["shape"], tf.uint8, name="images")]

        @tf.function(input_signature=signature)
        def wrapper(images):
            result = original(**{input_name: images})
            return tuple(result[n] for n in names)

        model, _ = tf2onnx.convert.from_function(
            wrapper, input_signature=signature, opset=18, output_path=str(target)
        )
        for i, (value, name) in enumerate(zip(model.graph.output, names)):
            if value.name != name:
                model.graph.node.append(
                    onnx.helper.make_node(
                        "Identity", [value.name], [name], name="output_alias_" + str(i)
                    )
                )
                value.name = name
        onnx.save_model(model, str(target))
        return references
    found = list(extracted.rglob("frozen_inference_graph.pb"))
    if len(found) != 1:
        raise ValueError("Expected one DeepLab frozen graph")
    graph_input = recipe["export"].get("graph_input", "sub_7:0")
    graph_output = recipe["export"].get("graph_output", "ResizeBilinear_2:0")
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(found[0].read_bytes())
    references = []
    size = recipe["preprocess"]["size"]
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")
        options = tf.compat.v1.ConfigProto(
            intra_op_parallelism_threads=1, inter_op_parallelism_threads=1
        )
        with tf.compat.v1.Session(graph=graph, config=options) as session:
            for feed in feeds:
                value = session.run(
                    "ResizeBilinear_3:0",
                    {
                        graph_input: feed["images"],
                        "ImageTensor:0": np.zeros((1, size, size, 3), np.uint8),
                    },
                )
                references.append(
                    {"logits": np.ascontiguousarray(value.transpose(0, 3, 1, 2))}
                )
    tf2onnx.convert.from_graph_def(
        graph_def,
        input_names=[graph_input],
        output_names=[graph_output],
        shape_override={graph_input: recipe["input"]["shape"]},
        opset=18,
        output_path=str(target),
    )
    from .tf_layout import tensorflow_final_graph, static_logits_resize

    model = tensorflow_final_graph(onnx.load(target), dict(input_size=size))
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    model = static_logits_resize(model)
    onnx.save_model(model, target)
    return references
