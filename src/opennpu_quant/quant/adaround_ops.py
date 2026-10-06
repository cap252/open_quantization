from onnx import helper


def attributes(node):
    values = {a.name: helper.get_attribute_value(a) for a in node.attribute}
    allowed = {
        "Conv": {"auto_pad", "dilations", "group", "kernel_shape", "pads", "strides"},
        "Gemm": {"alpha", "beta", "transA", "transB"},
        "MatMul": set(),
    }
    if node.domain not in ("", "ai.onnx") or node.op_type not in allowed:
        raise ValueError("Unsupported AdaRound operator: " + node.op_type)
    unknown = set(values) - allowed[node.op_type]
    if unknown:
        raise ValueError("Unsupported AdaRound attributes: " + str(sorted(unknown)))
    if node.op_type == "Gemm":
        if values.get("alpha", 1.0) != 1.0 or values.get("beta", 1.0) != 1.0:
            raise ValueError("AdaRound Gemm requires unit alpha/beta")
        if any(values.get(k, 0) not in (0, 1) for k in ("transA", "transB")):
            raise ValueError("Invalid Gemm transpose")
    return values


def operation(node, x, weight, bias):
    import torch
    import torch.nn.functional as F

    attrs = attributes(node)
    if node.op_type == "MatMul":
        if bias is not None:
            raise ValueError("MatMul cannot have a bias input")
        return torch.matmul(x, weight)
    if node.op_type == "Gemm":
        if x.ndim != 2 or weight.ndim != 2:
            raise ValueError("Gemm requires rank two")
        result = (x.T if attrs.get("transA", 0) else x) @ (
            weight.T if attrs.get("transB", 0) else weight
        )
        if bias is not None:
            if bias.shape != (result.shape[-1],):
                raise ValueError("Unsupported Gemm bias broadcasting")
            result = result + bias
        return result
    dims = weight.ndim - 2
    if dims not in (1, 2, 3) or x.ndim != dims + 2:
        raise ValueError("Conv supports one, two or three spatial dimensions")
    kernel = list(weight.shape[2:])
    if attrs.get("kernel_shape", kernel) != kernel:
        raise ValueError("Conv kernel_shape disagrees with weight")
    strides, dilations = (
        attrs.get("strides", [1] * dims),
        attrs.get("dilations", [1] * dims),
    )
    group = attrs.get("group", 1)
    if (
        group < 1
        or len(strides) != dims
        or len(dilations) != dims
        or min(*strides, *dilations) < 1
    ):
        raise ValueError("Invalid Conv stride/dilation/group")
    auto = attrs.get("auto_pad", b"NOTSET")
    if isinstance(auto, bytes):
        auto = auto.decode()
    pads = attrs.get("pads", [0] * (2 * dims))
    if auto in ("SAME_UPPER", "SAME_LOWER"):
        if "pads" in attrs:
            raise ValueError("Both auto_pad and pads specified")
        total = [
            max(
                0,
                ((x.shape[2 + i] + strides[i] - 1) // strides[i] - 1) * strides[i]
                + dilations[i] * (kernel[i] - 1)
                + 1
                - x.shape[2 + i],
            )
            for i in range(dims)
        ]
        before = [t // 2 if auto == "SAME_UPPER" else (t + 1) // 2 for t in total]
        pads = before + [t - p for t, p in zip(total, before)]
    elif auto == "VALID":
        if any(pads):
            raise ValueError("Conflicting Conv VALID padding")
        pads = [0] * (2 * dims)
    elif auto != "NOTSET":
        raise ValueError("Unsupported Conv auto_pad")
    if len(pads) != 2 * dims or min(pads) < 0:
        raise ValueError("Invalid Conv padding")
    if bias is not None and bias.shape != (weight.shape[0],):
        raise ValueError("Unsupported Conv bias broadcasting")
    padding = [v for i in reversed(range(dims)) for v in (pads[i], pads[dims + i])]
    if any(padding):
        x = F.pad(x, padding)
    if x.is_cuda:
        # Match ORT's cuDNN accumulation: native depthwise CUDA kernels can exceed
        # the fixed parity tolerance. Keep deterministic FP32 with TF32 disabled.
        cx = x.unsqueeze(2) if dims == 1 else x
        cw = weight.unsqueeze(2) if dims == 1 else weight
        cs = [1, *strides] if dims == 1 else strides
        cd = [1, *dilations] if dims == 1 else dilations
        result = torch.ops.aten.cudnn_convolution(
            cx, cw, [0] * max(2, dims), cs, cd, group, False, True, False
        )
        if dims == 1:
            result = result.squeeze(2)
        if bias is not None:
            result = result + bias.reshape(1, -1, *([1] * dims))
        return result
    return (F.conv1d, F.conv2d, F.conv3d)[dims - 1](
        x, weight, bias, stride=strides, dilation=dilations, groups=group
    )


def channel_axis(node):
    return 1 if node.op_type == "Conv" else -1
