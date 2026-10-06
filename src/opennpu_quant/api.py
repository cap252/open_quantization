def calibrate(model, feeds, *, config=None, ort=None):
    """Collect reusable activation statistics from a replayable input stream.

    Parameters
    ----------
    model : path-like or onnx.ModelProto
        Original FP32 network; paths resolve external weights inside its bundle.
    feeds : callable
        Zero-argument factory returning a fresh iterator of input-name-to-array
        dicts. Both passes must replay identical ordered values; lists, tuples
        and already-created generators are rejected.
    config : CalibrationConfig, optional
        Defaults to CalibrationConfig(); histogram_bins controls collection.
    ort : OrtConfig, optional
        Defaults to CPU. Calibration requires ORT_DISABLE_ALL.

    Returns
    -------
    CalibrationStats
        Min/max ranges, histograms and evidence for later range selection.
        The caller's model and feed arrays are not modified.

    Examples
    --------
    With a prepared model and a replayable feeds function (see the README):

    >>> from opennpu_quant import calibrate, OrtConfig
    >>> stats = calibrate(model, feeds, ort=OrtConfig.cpu())
    >>> assert stats.sample_count > 0
    """
    from .quant.calibration import calibrate as implementation
    from .quant.config import CalibrationConfig
    from .ort.config import OrtConfig

    return implementation(
        model, feeds, config=config or CalibrationConfig(), ort=ort or OrtConfig.cpu()
    )


def quantize(
    model,
    calibration,
    scheme=None,
    *,
    config=None,
    ort=None,
    reconstruction_feeds=None,
    checkpoint_dir=None,
):
    """Build an owned INT8 QDQ model from calibration feeds or statistics.

    Parameters
    ----------
    model : path-like or onnx.ModelProto
        Original network using ONNX opset 13 through 21. Feature inputs must
        be FP32; explicit INT64/boolean structural inputs are supported.
    calibration : callable or CalibrationStats
        Replayable feed factory, or unchanged statistics compatible with the
        model, histogram configuration and numerical ORT profile.
    scheme : Scheme, optional
        Compact policy; omitted scheme and config use Scheme() defaults.
    config : QuantizationConfig, optional
        Explicit policy, mutually exclusive with scheme. QuantizationConfig()
        has different defaults from Scheme(); neither is changed by this API.
    ort : OrtConfig, optional
        CPU by default; ORT_DISABLE_ALL is required.
    reconstruction_feeds : callable, optional
        AdaRound feed factory. Required with stored statistics when config
        enables AdaRound; otherwise the calibration factory can be reused.
    checkpoint_dir : path-like, optional
        Per-layer AdaRound checkpoint directory when config enables AdaRound.

    Returns
    -------
    QuantizationResult
        Owned model, effective configuration, range identity and audit.
        RTN runs first; config.adaround optionally optimizes weight codes.

    Examples
    --------
    With model and feeds prepared as in the README:

    >>> from opennpu_quant import quantize, QuantizationConfig
    >>> result = quantize(model, feeds, config=QuantizationConfig())
    >>> assert result.range_identity == result.audit['range_identity']
    """
    from .quant.api import quantize as implementation
    from .quant.scheme import Scheme
    from .ort.config import OrtConfig

    if scheme is not None and config is not None:
        raise ValueError("Specify scheme or config, not both")
    config = config if config is not None else (scheme or Scheme()).config()
    return implementation(
        model,
        calibration,
        config=config,
        ort=ort or OrtConfig.cpu(),
        reconstruction_feeds=reconstruction_feeds,
        adaround_checkpoint_dir=checkpoint_dir,
    )


def apply_adaround(
    fp32_model,
    quantized_result,
    feeds,
    config=None,
    *,
    ort=None,
    checkpoint_dir=None,
    activation_cache=None,
):
    """Optimize weight rounding codes while preserving the RTN encodings.

    Parameters
    ----------
    fp32_model : path-like or onnx.ModelProto
        Original network that produced quantized_result.
    quantized_result : QuantizationResult
        Verified RTN result; scales, zero points, bias and connections stay fixed.
    feeds : callable
        Zero-argument factory replaying identical ordered input dicts for each
        layer. Do not pass a list, tuple or an already-created generator.
    config : AdaroundConfig, optional
        Defaults to AdaroundConfig(), including 10000 steps per layer.
    ort : OrtConfig, optional
        CPU by default; ORT_DISABLE_ALL is required. Torch is required on use.
    checkpoint_dir : path-like, optional
        Save/resume a validated completed layer prefix. An interrupted layer
        restarts; incompatible or altered checkpoints are rejected.
    activation_cache : ActivationCacheConfig, optional
        Bounded intermediate activation cache. None disables this cache;
        this is independent of a calibration input disk cache.

    Returns
    -------
    QuantizationResult
        A new result with optimized codes and AdaRound audit metadata.
        The caller's original model and RTN result are not modified.

    Examples
    --------
    With model, feeds and an RTN result prepared first; reduced steps are a smoke check:

    >>> from opennpu_quant import apply_adaround, AdaroundConfig
    >>> options = AdaroundConfig(steps=20, batch_size=2, window_samples=4, window_steps=5)
    >>> optimized = apply_adaround(model, result, feeds, options)
    >>> assert optimized.range_identity == result.range_identity
    """
    from .quant.adaround import apply_adaround as implementation
    from .quant.config import AdaroundConfig
    from .ort.config import OrtConfig

    return implementation(
        fp32_model,
        quantized_result,
        feeds,
        config=config or AdaroundConfig(),
        ort=ort or OrtConfig.cpu(),
        checkpoint_dir=checkpoint_dir,
        activation_cache=activation_cache,
    )
