import collections


def summarize_profile(events):
    counts = collections.Counter()
    types = {}
    for event in events:
        a = event.get("args", {})
        if event.get("cat") != "Node" or not a.get("provider"):
            continue
        key = (a["provider"], a.get("op_name", "unknown"))
        counts[key] += 1
        types.setdefault(
            key,
            {
                "input_type_shape": a.get("input_type_shape"),
                "output_type_shape": a.get("output_type_shape"),
            },
        )
    rows = [
        {"provider": p, "op_type": o, "calls": n, **types[(p, o)]}
        for (p, o), n in sorted(counts.items())
    ]
    cuda = sum(r["calls"] for r in rows if r["provider"] == "CUDAExecutionProvider")
    cpu = sum(r["calls"] for r in rows if r["provider"] == "CPUExecutionProvider")
    integer = sum(
        r["calls"]
        for r in rows
        if r["provider"] == "CUDAExecutionProvider"
        and r["op_type"].startswith(
            ("QLinearConv", "QLinearMatMul", "QGemm", "ConvInteger", "MatMulInteger")
        )
    )
    return {
        "operators": rows,
        "cuda_calls": cuda,
        "cpu_calls": cpu,
        "cuda_integer_compute_calls": integer,
        "cuda_fraction_by_node_calls": cuda / (cuda + cpu) if cuda + cpu else 0.0,
    }


def assert_cuda_observed(providers, profile):
    if "CUDAExecutionProvider" not in providers or not profile["cuda_calls"]:
        raise RuntimeError(
            "CUDA was requested but no CUDA kernel executed; CPU fallback is not GPU success"
        )
    if not any(
        r["provider"] == "CUDAExecutionProvider"
        and r["op_type"]
        in (
            "Conv",
            "FusedConv",
            "NhwcConv",
            "MatMul",
            "Gemm",
            "FusedMatMul",
            "QLinearConv",
            "QLinearMatMul",
        )
        for r in profile["operators"]
    ):
        raise RuntimeError("No main neural-network computation was observed on CUDA")


def profile_sample(model, feeds, expected, config):
    """Profile one real input; do not confuse configured providers with execution."""
    import json, tempfile
    from pathlib import Path
    import numpy as np
    from .session import create_session
    from .fetch import run_tensors

    with tempfile.TemporaryDirectory(
        dir=config.scratch, prefix="opennpu_profile_"
    ) as folder:
        session = create_session(model, config, profile=Path(folder) / "profile")
        names = list(expected)
        outputs = run_tensors(session, names, feeds)
        path = Path(session.end_profiling())
        for name, output in zip(names, outputs):
            np.testing.assert_allclose(output, expected[name], atol=1e-5, rtol=1e-4)
        summary = summarize_profile(json.loads(path.read_text()))
        if "CUDAExecutionProvider" in config.providers:
            assert_cuda_observed(session.get_providers(), summary)
        return summary
