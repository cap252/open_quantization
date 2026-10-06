import numpy as np


class TopKClassification:
    """Accumulate per-image Top-K classification accuracy for evaluate().

    Parameters
    ----------
    output_name : str
        Logit tensor key, default 'logits'; must have shape [1, classes].
    topk : iterable of int
        Nonempty positive K values, default (1, 5); logits need at least max(K)
        classes. EvaluationSample.target is the zero-based class index.

    reset() clears counts, update(outputs, sample) adds one observation and
    finalize() returns fraction-valued topK metrics. Empty observations and
    nonfinite logits are rejected. This adds no original-framework parity claim.

    Examples
    --------
    >>> from opennpu_quant import TopKClassification
    >>> evaluator = TopKClassification(output_name='logits', topk=(1, 5))
    >>> assert evaluator.metric_units == {'top1': 'fraction', 'top5': 'fraction'}
    """

    def __init__(self, *, output_name="logits", topk=(1, 5)):
        if not topk or any((k < 1 for k in topk)):
            raise ValueError("Positive topk required")
        self.output_name = output_name
        self.topk = tuple(topk)
        self.reset()

    def reset(self):
        self.count = 0
        self.prediction = None
        self.correct = {k: 0 for k in self.topk}

    def update(self, outputs, sample):
        logits = np.asarray(outputs[self.output_name])
        if logits.ndim != 2 or logits.shape[0] != 1 or logits.shape[1] < max(self.topk):
            raise ValueError(
                "One sample requires [1, classes] logits with sufficient classes"
            )
        if (
            not np.isfinite(logits).all()
            or not 0 <= int(sample.target) < logits.shape[1]
        ):
            raise ValueError("Invalid logits or target")
        prediction = np.argsort(logits[0])[::-1]
        for k in self.topk:
            self.correct[k] += int(int(sample.target) in prediction[:k])
        self.count += 1
        self.prediction = dict(
            image_id=sample.sample_id,
            label=int(sample.target),
            top5=prediction[:5].tolist(),
            top5_logits=logits[0, prediction[:5]].tolist(),
        )

    @property
    def metric_units(self):
        return {f"top{k}": "fraction" for k in self.topk}

    def finalize(self):
        if not self.count:
            raise ValueError("No classification observations")
        return {f"top{k}": v / self.count for k, v in self.correct.items()}


class SegmentationEvaluator:
    def __init__(self, *, output_name, classes=21, ignore_label=255):
        self.output_name = output_name
        self.classes = classes
        self.ignore_label = ignore_label
        self.reset()

    def reset(self):
        from .segmentation import _ConfusionMatrix

        self._state = _ConfusionMatrix(self.classes, self.ignore_label)
        self.metric_details = {}

    @property
    def confusion(self):
        return self._state.matrix

    @confusion.setter
    def confusion(self, value):
        self._state.matrix = value

    @property
    def metric_units(self):
        return {"mIoU": "fraction"}

    def update(self, outputs, sample):
        logits = np.asarray(outputs[self.output_name])
        if logits.ndim == 4:
            if logits.shape[:2] != (1, self.classes) or not np.isfinite(logits).all():
                raise ValueError(
                    "Expected finite [1, classes, H, W] segmentation logits"
                )
            predicted = logits.argmax(1)[0]
        else:
            predicted = logits
        self._state.add(predicted, sample.target)

    def finalize(self):
        result = self._state.metrics()
        self.metric_details = {k: v for k, v in result.items() if k != "mIoU"}
        return {"mIoU": result["mIoU"]}


class DecodedEvaluator:
    """Adapter-owned decoder with a reusable metric accumulator."""

    def __init__(self, decoder, metric):
        self.decoder, self.metric = (decoder, metric)

    def reset(self):
        self.metric.reset()

    def update(self, outputs, sample):
        self.metric.update(self.decoder(outputs, sample.metadata), sample)

    def finalize(self):
        return self.metric.finalize()

    @property
    def prediction(self):
        return getattr(self.metric, "prediction", None)

    @property
    def metric_details(self):
        return getattr(
            self.metric, "metric_details", getattr(self.metric, "details", {})
        )

    @property
    def metric_units(self):
        return getattr(self.metric, "metric_units", {})
