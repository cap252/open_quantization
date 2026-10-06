from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterator, Mapping, Union

if TYPE_CHECKING:
    import numpy as np
    import onnx
OnnxSource = Union[str, Path, "onnx.ModelProto"]
FeedFactory = Callable[[], Iterator[Mapping[str, "np.ndarray"]]]
