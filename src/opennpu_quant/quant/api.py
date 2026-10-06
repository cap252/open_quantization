from dataclasses import dataclass, replace
from opennpu_quant.graph.model import load_model, model_identity
from .calibration import CalibrationStats, calibrate, validate_statistics
from .config import QuantizationConfig
from opennpu_quant.models.types import OnnxSource, FeedFactory
from opennpu_quant.ort.config import OrtConfig


@dataclass(frozen=True)
class QuantizationResult:
    """Owned QDQ model and evidence returned by quantize() or apply_adaround().

    Attributes
    ----------
    model : onnx.ModelProto
        Result graph; the caller's original model is not modified.
    config : QuantizationConfig
        Effective quantization policy. RTN construction clears config.adaround;
        optimization then sets config.adaround and records audit['adaround'].
    range_identity : str
        Selected range/encoding identity, preserved by fixed-encoding AdaRound.
    audit : dict
        Source-model identity, encoded-parameter/protection checks and, when
        optimized, AdaRound details. Do not edit evidence to force reuse.

    Examples
    --------
    With a prepared model and feeds:

    >>> from opennpu_quant import quantize, QuantizationConfig
    >>> result = quantize(model, feeds, config=QuantizationConfig())
    >>> assert result.range_identity == result.audit['range_identity']
    """

    model: object
    config: QuantizationConfig
    range_identity: str
    audit: dict


def quantize(
    model: OnnxSource,
    calibration: FeedFactory | CalibrationStats,
    *,
    config: QuantizationConfig,
    ort: OrtConfig,
    reconstruction_feeds: FeedFactory | None = None,
    adaround_checkpoint_dir=None,
) -> QuantizationResult:
    from .engine import quantize_model
    from .ranges import select_ranges

    reconstruction = (
        reconstruction_feeds
        if reconstruction_feeds is not None
        else (calibration if callable(calibration) else None)
    )
    if config.adaround is not None and reconstruction is None:
        raise ValueError("AdaRound with CalibrationStats requires reconstruction_feeds")

    if ort.optimization != "ORT_DISABLE_ALL":
        raise ValueError("QDQ requires ORT_DISABLE_ALL")
    owned = load_model(model)
    if any((v.type.tensor_type.elem_type not in (1, 7, 9) for v in owned.graph.input)):
        raise ValueError(
            "Only FP32 features and explicit INT64/boolean structural inputs are supported"
        )
    opset = next(
        (v.version for v in owned.opset_import if v.domain in ("", "ai.onnx")), 0
    )
    if not 13 <= opset <= 21:
        raise ValueError("QDQ supports verified ONNX opsets 13 through 21")
    stats = (
        calibration
        if isinstance(calibration, CalibrationStats)
        else calibrate(owned, calibration, config=config.calibration, ort=ort)
    )
    validate_statistics(owned, stats, config=config.calibration, ort=ort)
    ranges = select_ranges(
        stats.histograms,
        stats.minmax,
        config=config.calibration,
        activation_symmetric=config.activation_symmetric,
    )
    rtn_config = replace(config, adaround=None)
    result, audit = quantize_model(owned, ranges, rtn_config.scheme(), stats.protection)
    audit["source_model_identity"] = model_identity(owned)
    value = QuantizationResult(result, rtn_config, audit["range_identity"], audit)
    if config.adaround is not None:
        from .adaround import apply_adaround

        return apply_adaround(
            owned,
            value,
            reconstruction,
            config=config.adaround,
            ort=ort,
            checkpoint_dir=adaround_checkpoint_dir,
        )
    return value
