from dataclasses import dataclass
from pathlib import Path
import hashlib
import tempfile
import numpy as np
import onnx
from opennpu_quant._io import object_hash
from opennpu_quant.graph.model import model_identity
from opennpu_quant.ort.session import create_session, environment
from opennpu_quant.ort.fetch import run_tensors
from .config import CalibrationConfig
from opennpu_quant.models.types import OnnxSource, FeedFactory
from opennpu_quant.ort.config import OrtConfig
from .histograms import FixedHistograms


from opennpu_quant._feeds import feed_digest


@dataclass(frozen=True)
class CalibrationStats:
    """Collected ranges, histograms and reuse evidence returned by calibrate().

    Attributes
    ----------
    model_identity, plan_identity, samples_identity : str
        Original model, histogram plan and ordered input fingerprints.
    sample_count, histogram_bins : int
        Collected feed count and bin count.
    minmax, histograms : dict, FixedHistograms
        Tensor ranges and signed/absolute histogram observations.
    protection : dict
        Activation boundaries used for graph and tensor coverage validation.
    environment : dict
        Numerical ORT collection profile; physical paths are not compatibility proof.
    statistics_identity : str
        Digest of the numerical ranges and histogram content.

    Use calibrate() rather than constructing evidence manually. The dataclass is
    frozen, but nested arrays/dicts must also remain unchanged. quantize() validates
    model, plan, histograms, environment and statistics before reuse.

    Examples
    --------
    With model and a replayable feeds function:

    >>> from opennpu_quant import calibrate, quantize, QuantizationConfig
    >>> stats = calibrate(model, feeds)
    >>> reused = quantize(model, stats, config=QuantizationConfig())
    >>> assert stats.sample_count > 0
    """

    model_identity: str
    plan_identity: str
    samples_identity: str
    sample_count: int
    histogram_bins: int
    minmax: dict
    histograms: FixedHistograms
    protection: dict
    environment: dict
    statistics_identity: str


def statistics_identity(ranges, histograms):
    digest = hashlib.sha256(object_hash(ranges).encode())
    for name, row in sorted(histograms.rows.items()):
        digest.update(name.encode())
        for key in ("signed", "absolute", "signed_edges", "absolute_edges"):
            digest.update(np.ascontiguousarray(row[key]).tobytes())
        digest.update(object_hash([int(row["count"]), int(row["images"])]).encode())
    return digest.hexdigest()


def collect_statistics(
    source, feeds, *, bins, session_factory, scratch=None, execution=None
):
    from .engine import infer_model, layout, _add_calibration_outputs

    if not callable(feeds):
        raise TypeError(
            "calibrate() needs a fresh-iterator feed factory: a zero-argument callable "
            "returning a new iterator of {input name: array} dicts on every call "
            "(calibration reads inputs twice). Example: calibrate(model, lambda: iter(feed_list))"
        )
    model = infer_model(source)
    identity = model_identity(source)
    plan = layout(model, "all")
    names = plan["tensors"]
    _add_calibration_outputs(model, names)
    ranges = {name: dict(lowest=float("inf"), highest=-float("inf")) for name in names}
    signatures = []
    sample_digests = []
    counts = []
    with tempfile.TemporaryDirectory(
        dir=scratch, prefix="opennpu_calibration_"
    ) as directory:
        path = Path(directory) / "augmented.onnx"
        onnx.save_model(model, path)
        session = session_factory(path)
        for pass_index in range(2):
            if pass_index:
                histograms = FixedHistograms(ranges, bins)
            signature = hashlib.sha256()
            count = 0
            for feed in feeds():
                before = feed_digest(feed)
                if not pass_index:
                    sample_digests.append(before)
                elif count >= len(sample_digests) or sample_digests[count] != before:
                    raise ValueError(
                        "Calibration replay changed ordered samples or values"
                    )
                values = run_tensors(session, names, feed)
                if feed_digest(feed) != before:
                    raise ValueError("Calibration modified caller inputs")
                signature.update(before.encode())
                for name, value in zip(names, values):
                    if (
                        value.dtype != np.float32
                        or not value.size
                        or (not np.isfinite(value).all())
                    ):
                        raise ValueError("Invalid FP32 calibration tensor: " + name)
                    if pass_index:
                        histograms.add(name, value)
                    else:
                        ranges[name]["lowest"] = min(
                            ranges[name]["lowest"], float(value.min())
                        )
                        ranges[name]["highest"] = max(
                            ranges[name]["highest"], float(value.max())
                        )
                count += 1
            signatures.append(signature.hexdigest())
            counts.append(count)
        del session
    if counts[0] == 0 or counts[0] != counts[1] or signatures[0] != signatures[1]:
        raise ValueError("Calibration replay changed ordered samples or values")
    return CalibrationStats(
        identity,
        object_hash(plan["rows"]),
        signatures[0],
        counts[0],
        bins,
        ranges,
        histograms,
        plan["protection"],
        execution or {},
        statistics_identity(ranges, histograms),
    )


def calibrate(
    model: OnnxSource, feeds: FeedFactory, *, config: CalibrationConfig, ort: OrtConfig
) -> CalibrationStats:
    """Require feeds() to replay identical ordered inputs for both passes."""
    if ort.optimization != "ORT_DISABLE_ALL":
        raise ValueError(
            "Calibration/QDQ require ORT_DISABLE_ALL to preserve protected boundaries"
        )
    return collect_statistics(
        model,
        feeds,
        bins=config.histogram_bins,
        session_factory=lambda source: create_session(source, ort),
        scratch=ort.scratch,
        execution=environment(ort),
    )


def validate_statistics(model, stats, *, config, ort, samples_identity=None):
    """Validate an imported bundle against its consumer, never its original path."""
    from .engine import infer_model, layout

    if not isinstance(stats, CalibrationStats):
        raise ValueError("Expected CalibrationStats")
    if stats.model_identity != model_identity(model):
        raise ValueError("Calibration model identity mismatch")
    if (
        stats.histogram_bins != config.histogram_bins
        or stats.histograms.bins != stats.histogram_bins
    ):
        raise ValueError("Calibration histogram configuration mismatch")
    if stats.environment != environment(ort):
        raise ValueError("Calibration ORT environment mismatch")
    if samples_identity is not None and stats.samples_identity != samples_identity:
        raise ValueError("Calibration ordered feed identity mismatch")
    plan = layout(infer_model(model), "all")
    if object_hash(stats.protection) != object_hash(plan["protection"]):
        raise ValueError("Calibration protection manifest was modified")
    if stats.plan_identity != object_hash(plan["rows"]):
        raise ValueError("Calibration role/protection plan mismatch")
    if set(stats.minmax) != set(plan["tensors"]) or set(stats.histograms.rows) != set(
        plan["tensors"]
    ):
        raise ValueError("Calibration tensor coverage mismatch")
    if stats.sample_count < 1 or any(
        r["images"] != stats.sample_count for r in stats.histograms.rows.values()
    ):
        raise ValueError("Calibration observation count mismatch")
    if stats.statistics_identity != statistics_identity(stats.minmax, stats.histograms):
        raise ValueError("Calibration statistics were modified")
    expected = FixedHistograms(stats.minmax, stats.histogram_bins)
    for name, row in stats.histograms.rows.items():
        shape = expected.rows[name]
        if any(row[key] != shape[key] for key in ("low", "high", "bound")):
            raise ValueError("Calibration histogram range metadata mismatch")
        for key in ("signed_edges", "absolute_edges"):
            if not np.array_equal(row[key], shape[key]):
                raise ValueError("Calibration histogram edges mismatch")
        for key in ("signed", "absolute"):
            counts = np.asarray(row[key])
            if (
                counts.shape != (stats.histogram_bins,)
                or counts.dtype != np.int64
                or np.any(counts < 0)
                or int(counts.sum()) != row["count"]
            ):
                raise ValueError("Calibration histogram counts mismatch")


def load_statistics(folder):
    """Load an explicit bundle; semantic validation is required before consumption."""
    from opennpu_quant._io import read_json

    folder = Path(folder)
    return CalibrationStats(
        histograms=FixedHistograms.load(folder / "histograms.npz"),
        **read_json(folder / "statistics.json"),
    )
