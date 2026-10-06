from .config import OrtConfig


def runtime(config=None):
    import onnxruntime as ort

    policy = config if config is not None else OrtConfig.cpu()
    if ort.__version__ not in policy.allowed_versions:
        message = f"Unverified ORT version {ort.__version__}; reference: {policy.allowed_versions}"
        if policy.strict_versions:
            raise RuntimeError(message)
        import warnings

        warnings.warn(message, RuntimeWarning, stacklevel=2)
    return ort


def create_session(model, config, *, profile=None, optimized=None):
    ort = runtime(config)
    missing = set(config.providers) - set(ort.get_available_providers())
    if missing:
        raise RuntimeError(
            "Requested ORT providers unavailable: " + ", ".join(sorted(missing))
        )
    if "CUDAExecutionProvider" in config.providers:
        from .cuda_libs import load_cuda_libraries

        load_cuda_libraries()
    options = ort.SessionOptions()
    options.intra_op_num_threads = config.intra_op_threads
    options.inter_op_num_threads = config.inter_op_threads
    options.graph_optimization_level = getattr(
        ort.GraphOptimizationLevel, config.optimization
    )
    options.log_severity_level = 3
    if profile is not None:
        options.enable_profiling = True
        options.profile_file_prefix = str(profile)
    if optimized is not None:
        options.optimized_model_filepath = str(optimized)
    providers = (
        [
            (name, dict(config.provider_options[i]))
            for i, name in enumerate(config.providers)
        ]
        if config.provider_options
        else list(config.providers)
    )
    if hasattr(model, "SerializeToString"):
        from opennpu_quant.graph.model import load_model

        model = load_model(model).SerializeToString()
    argument = model if isinstance(model, bytes) else str(model)
    result = ort.InferenceSession(argument, options, providers=providers)
    result.disable_fallback()
    if any((p not in result.get_providers() for p in config.providers)):
        raise RuntimeError(
            "ORT provider initialization failed; implicit fallback is disabled"
        )
    return result


def environment(config):
    """Numerical profile for reuse; physical paths, UUIDs and host names are excluded."""
    import numpy, onnx

    ort = runtime(config)
    options = [
        {k: v for k, v in row if k not in ("device_id", "gpu_mem_limit")}
        for row in config.provider_options
    ]
    return dict(
        onnxruntime=ort.__version__,
        onnx=onnx.__version__,
        numpy=numpy.__version__,
        providers=list(config.providers),
        provider_options=options,
        optimization=config.optimization,
        intra_op_threads=config.intra_op_threads,
        inter_op_threads=config.inter_op_threads,
    )
