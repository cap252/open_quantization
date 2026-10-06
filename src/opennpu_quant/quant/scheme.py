from dataclasses import dataclass, asdict
from .config import CalibrationConfig, QuantizationConfig


@dataclass(frozen=True)
class Scheme:
    """Compact immutable policy converted to QuantizationConfig by config().

    Attributes
    ----------
    scope : str
        'basic' or 'all' (default).
    activation_symmetric : bool
        False by default; activation is signed INT8 per-tensor.
    weight_granularity : str
        'per_channel' (default) or 'per_tensor'; weights are symmetric INT8.
    method : str
        'percentile' (default), 'minmax' or 'entropy'.
    percentile : float or None
        Default 99.99 in (0, 100]; must be None for minmax or entropy.
    activation_scale, weight_scale : str
        Default 'float'; also 'pot_nearest' and 'pot_ceil'. Weight alone also
        supports 'pot_mse'. These describe INT8 scales, not FP32 execution.
    histogram_bins, quantized_bins : int
        Defaults 2048 and 128; 2 <= quantized_bins <= histogram_bins.

    Scheme() defaults differ from QuantizationConfig(); conversion is explicit.

    Examples
    --------
    >>> from opennpu_quant import Scheme
    >>> scheme = Scheme(activation_scale='pot_ceil', weight_scale='pot_ceil')
    >>> config = scheme.config()
    >>> assert config.weight_granularity == 'per_channel'
    """

    scope: str = "all"
    activation_symmetric: bool = False
    weight_granularity: str = "per_channel"
    method: str = "percentile"
    percentile: float | None = 99.99
    activation_scale: str = "float"
    weight_scale: str = "float"
    histogram_bins: int = 2048
    quantized_bins: int = 128

    def __post_init__(self):
        self.config()

    def config(self):
        methods = {"percentile": "Percentile", "minmax": "MinMax", "entropy": "Entropy"}
        if self.method not in methods:
            raise ValueError("Unknown calibration method: " + self.method)
        if self.method != "percentile" and self.percentile is not None:
            raise ValueError("Use percentile=None with minmax or entropy")
        return QuantizationConfig(
            scope=self.scope,
            activation_symmetric=self.activation_symmetric,
            weight_granularity=self.weight_granularity,
            activation_scale_policy=self.activation_scale,
            weight_scale_policy=self.weight_scale,
            calibration=CalibrationConfig(
                methods[self.method],
                self.percentile,
                self.histogram_bins,
                self.quantized_bins,
            ),
        )

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        return cls(**value)

    @property
    def name(self):
        return self.config().name
