from dataclasses import dataclass, field
from copy import deepcopy
import time
import numpy as np
from ..ort.config import OrtConfig
from ..ort.session import create_session, environment
from ..ort.fetch import run_tensors


@dataclass(frozen=True)
class EvaluationResult:
    metrics: dict
    samples: int
    complete: bool
    elapsed_s: float
    runtime: dict
    metric_details: dict = field(default_factory=dict)
    metric_units: dict = field(default_factory=dict)


def evaluate(model, samples, *, evaluator, ort=None, expected_samples=None, limit=None):
    """Stream samples through ONNX Runtime and a caller-supplied evaluator.

    Parameters
    ----------
    model : path-like or onnx.ModelProto
        Network to evaluate with the supplied input dictionaries.
    samples : iterable of EvaluationSample
        One pass of uniquely identified, preprocessed samples; not a factory.
    evaluator : object
        Must implement reset(), update(outputs, sample), and finalize().
        finalize returns finite scalar metrics; metric_units and metric_details
        are optional. This function does not choose preprocessing or a metric.
    ort : OrtConfig, optional
        CPU by default; controls execution and optional placement profiling.
    expected_samples : int, optional
        Full evaluation count. Missing requested samples raise an error.
    limit : int, optional
        Positive maximum to consume. A limited subset is not a complete result.

    Returns
    -------
    EvaluationResult
        Metrics, sample count, elapsed time, runtime, units and details.
        complete is true only when expected_samples is given and reached.
        Empty/duplicate samples or nonfinite outputs/metrics are rejected.

    Examples
    --------
    With model and feeds prepared first:

    >>> from opennpu_quant import evaluate, EvaluationSample, TopKClassification
    >>> items = [EvaluationSample(str(i), feed, 5) for i, feed in enumerate(feeds())]
    >>> measured = evaluate(model, items, evaluator=TopKClassification(), expected_samples=len(items))
    >>> assert measured.complete
    """
    ort = ort or OrtConfig.cpu()
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("limit must be positive")
    session = create_session(model, ort)
    names = [v.name for v in session.get_outputs()]
    seen = set()
    evaluator.reset()
    start = time.monotonic()
    placement = None
    from itertools import islice

    for sample in islice(iter(samples), limit):
        if sample.sample_id in seen:
            raise ValueError("Duplicate sample: " + str(sample.sample_id))
        seen.add(sample.sample_id)
        values = run_tensors(session, names, sample.feeds)
        if any(not np.isfinite(v).all() for v in values):
            raise ValueError("Nonfinite prediction")
        if placement is None and ort.profile_samples:
            from ..ort.profile import profile_sample

            placement = profile_sample(
                model, sample.feeds, dict(zip(names, values)), ort
            )
        evaluator.update(dict(zip(names, values)), sample)
    if not seen:
        raise ValueError("No evaluation samples")
    requested = (
        min(expected_samples, limit)
        if expected_samples is not None and limit
        else expected_samples
    )
    if requested is not None and len(seen) != requested:
        raise ValueError("Missing evaluation samples")
    metrics = dict(evaluator.finalize())
    if any(not np.isfinite(v) for v in metrics.values()):
        raise ValueError("Nonfinite metric")
    return EvaluationResult(
        metrics,
        len(seen),
        expected_samples is not None and len(seen) == expected_samples,
        time.monotonic() - start,
        dict(**environment(ort), placement=placement),
        deepcopy(getattr(evaluator, "metric_details", {})),
        {
            key: getattr(evaluator, "metric_units", {}).get(key, "fraction")
            for key in metrics
        },
    )
