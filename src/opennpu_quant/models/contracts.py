from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class EvaluationSample:
    """One preprocessed evaluation input and its task-specific target.

    Attributes
    ----------
    sample_id : str
        Unique identifier within an evaluate() call; duplicates are rejected.
    feeds : mapping
        Network input names to arrays already prepared for the model.
    target : object
        Ground truth interpreted by the chosen evaluator (e.g. a class index).
    metadata : mapping
        Optional task-specific information such as original size; defaults empty.

    The frozen record does not copy or freeze nested arrays. Evaluation consumes
    an iterable of these records once, unlike calibration's replayable factory.

    Examples
    --------
    >>> import numpy as np
    >>> from opennpu_quant import EvaluationSample
    >>> sample = EvaluationSample('image-0', {'images': np.zeros((1, 4), np.float32)}, 5)
    >>> assert sample.target == 5
    """

    sample_id: str
    feeds: Mapping[str, Any]
    target: Any
    metadata: Mapping[str, Any] = field(default_factory=dict)
