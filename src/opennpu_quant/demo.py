def run(*, adaround=False, device="cpu"):
    import numpy as np
    from onnx import helper as h, numpy_helper as n
    from . import (
        OrtConfig,
        Scheme,
        AdaroundConfig,
        ActivationCacheConfig,
        calibrate,
        quantize,
        apply_adaround,
        evaluate,
        EvaluationSample,
        TopKClassification,
    )

    model = h.make_model(
        h.make_graph(
            [
                h.make_node(
                    "Gemm", ["images", "w", "b"], ["hidden"], name="projection"
                ),
                h.make_node("Sigmoid", ["hidden"], ["gate"], name="gate"),
                h.make_node("Mul", ["hidden", "gate"], ["logits"], name="silu"),
            ],
            "tiny",
            [h.make_tensor_value_info("images", 1, [1, 4])],
            [h.make_tensor_value_info("logits", 1, [1, 6])],
            [
                n.from_array(
                    np.arange(24, dtype=np.float32).reshape(4, 6) / 40 - 0.3, "w"
                ),
                n.from_array(np.arange(6, dtype=np.float32) / 10, "b"),
            ],
        ),
        opset_imports=[h.make_opsetid("", 18)],
        ir_version=10,
    )

    def feeds():
        for i in range(8):
            yield {"images": np.asarray([[i / 8, -0.5, 0.3, 0.1]], np.float32)}

    def samples():
        for i, f in enumerate(feeds()):
            yield EvaluationSample(str(i), f, 5)

    ort = (
        OrtConfig.cpu()
        if device == "cpu"
        else OrtConfig.cuda(int(device.split(":")[1]))
    )
    stats = calibrate(model, feeds, ort=ort)
    records = []
    baseline = evaluate(
        model, samples(), evaluator=TopKClassification(), ort=ort, expected_samples=8
    )
    for family, scale in [("float", "float"), ("pot", "pot_ceil")]:
        q = quantize(
            model, stats, Scheme(activation_scale=scale, weight_scale=scale), ort=ort
        )
        for rounding in ["rtn"] + (["adaround"] if adaround else []):
            if rounding == "adaround":
                q = apply_adaround(
                    model,
                    q,
                    feeds,
                    AdaroundConfig(
                        steps=20, batch_size=2, window_samples=4, window_steps=5
                    ),
                    ort=ort,
                    activation_cache=ActivationCacheConfig(host_bytes=1024**2),
                )
            result = evaluate(
                q.model,
                samples(),
                evaluator=TopKClassification(),
                ort=ort,
                expected_samples=8,
            )
            records.append(
                dict(
                    condition=family + "_" + rounding,
                    metrics=result.metrics,
                    internal_qdq_count=q.audit["internal_qdq_count"],
                )
            )
    return dict(
        kind="synthetic API smoke, not pretrained accuracy",
        fp32=baseline.metrics,
        results=records,
    )
