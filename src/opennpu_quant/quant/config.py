from dataclasses import asdict, dataclass, field

ACTIVATION_SCALE_POLICIES = ("float", "pot_nearest", "pot_ceil")
WEIGHT_SCALE_POLICIES = (*ACTIVATION_SCALE_POLICIES, "pot_mse")


def scale_policies(scheme):
    activation = scheme.get("activation_scale_policy", "float")
    weight = scheme.get("weight_scale_policy", "float")
    if (
        activation not in ACTIVATION_SCALE_POLICIES
        or weight not in WEIGHT_SCALE_POLICIES
    ):
        raise ValueError("Unsupported activation/weight scale policy")
    return activation, weight


@dataclass(frozen=True)
class QuantPrecision:
    bits: int = 8
    signed: bool = True


@dataclass(frozen=True)
class CalibrationConfig:
    """Immutable histogram collection and activation range selection policy.

    Attributes
    ----------
    method : str
        'MinMax' (default), 'Percentile' or 'Entropy'. Collection records min/max
        and histograms; the method selects ranges when quantizing.
    percentile_value : float or None
        Required in (0, 100] for Percentile; None for other methods.
    histogram_bins, quantized_bins : int
        Defaults 2048 and 128; 2 <= quantized_bins <= histogram_bins.

    percentile(value=99.999) and entropy() are convenience constructors.

    Examples
    --------
    >>> from opennpu_quant import CalibrationConfig
    >>> selection = CalibrationConfig.percentile(99.99)
    >>> assert selection.method == 'Percentile'
    """

    method: str = "MinMax"
    percentile_value: float | None = None
    histogram_bins: int = 2048
    quantized_bins: int = 128

    def __post_init__(self):
        if self.method not in ("MinMax", "Percentile", "Entropy"):
            raise ValueError("Unsupported calibration method")
        if self.method == "Percentile":
            if self.percentile_value is None or not 0 < self.percentile_value <= 100:
                raise ValueError("Percentile must be in (0, 100]")
        elif self.percentile_value is not None:
            raise ValueError("percentile_value is only valid for Percentile")
        if (
            self.histogram_bins < 2
            or not 2 <= self.quantized_bins <= self.histogram_bins
        ):
            raise ValueError("Invalid histogram/quantized bin count")

    @classmethod
    def percentile(cls, value=99.999, **kwargs):
        return cls("Percentile", value, **kwargs)

    @classmethod
    def entropy(cls, **kwargs):
        return cls("Entropy", **kwargs)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        return cls(**value)


@dataclass(frozen=True)
class AdaroundConfig:
    """Immutable fixed-encoding weight rounding settings; Torch loads on use.

    Attributes
    ----------
    steps, learning_rate : int, float
        Positive steps per layer and learning rate; defaults 10000 and 1e-3.
    warmup_fraction, regularization : float
        Defaults 0.2 and 0.01; warmup is in [0, 1), regularization is nonnegative.
    beta_start, beta_end : float
        Defaults 20 and 2; beta_start >= beta_end > 0.
    seed : int
        Nonnegative deterministic layer seed base; default 20260928.
    batch_size, window_samples, window_bytes, window_steps : int
        Positive limits; defaults 8, 32, 512 MiB and 100. Byte limits are bytes.
    device : str
        'auto' (default), 'cpu' or 'cuda'; auto follows the ORT providers.
        GPU index comes from OrtConfig, not a 'cuda:N' value here.
    atol, rtol : float
        Nonnegative local-operation parity bounds, defaults 1e-5 and 1e-4.
        All floating settings must be finite. Do not loosen bounds to hide failures.

    Examples
    --------
    >>> from opennpu_quant import AdaroundConfig
    >>> smoke = AdaroundConfig(steps=20, batch_size=2, window_samples=4, window_steps=5)
    >>> assert smoke.steps == 20  # reduced smoke run, not the reference protocol
    """

    steps: int = 10000
    learning_rate: float = 1e-3
    warmup_fraction: float = 0.2
    regularization: float = 0.01
    beta_start: float = 20.0
    beta_end: float = 2.0
    seed: int = 20260928
    batch_size: int = 8
    window_samples: int = 32
    window_bytes: int = 512 * 1024**2
    window_steps: int = 100
    device: str = "auto"
    atol: float = 1e-5
    rtol: float = 1e-4

    def __post_init__(self):
        import math

        for key in (
            "steps",
            "batch_size",
            "window_samples",
            "window_bytes",
            "window_steps",
        ):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError("AdaRound requires a positive integer: " + key)
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("AdaRound seed must be a nonnegative integer")
        values = (
            self.learning_rate,
            self.regularization,
            self.beta_start,
            self.beta_end,
            self.warmup_fraction,
            self.atol,
            self.rtol,
        )
        if not all(math.isfinite(v) for v in values):
            raise ValueError("Nonfinite AdaRound configuration")
        if (
            self.learning_rate <= 0
            or self.regularization < 0
            or not 0 <= self.warmup_fraction < 1
        ):
            raise ValueError("Invalid AdaRound learning/loss policy")
        if (
            self.beta_start < self.beta_end
            or self.beta_end <= 0
            or self.atol < 0
            or self.rtol < 0
        ):
            raise ValueError("Invalid AdaRound beta/parity policy")
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError("AdaRound device must be auto/cpu/cuda")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        return cls(**value)

    @property
    def identity(self):
        from opennpu_quant._io import object_hash

        return object_hash(self.to_dict())


@dataclass(frozen=True)
class QuantizationConfig:
    """Immutable explicit INT8 QDQ policy; defaults preserve the basic API path.

    Attributes
    ----------
    activation_precision, weight_precision : QuantPrecision
        Signed 8-bit only; defaults instantiate that supported precision.
    activation_symmetric, weight_symmetric : bool
        Both default True; asymmetric activation is supported, weights stay symmetric.
    activation_granularity, weight_granularity : str
        Activation is 'per_tensor'; weight is 'per_tensor' (default) or 'per_channel'.
    scope, representation : str
        Scope is 'basic' (default) or 'all'; representation is 'QDQ'.
    calibration : CalibrationConfig
        Defaults to MinMax. Scheme() instead defaults to all-scope asymmetric
        activation, per-channel weights and Percentile; defaults are not unified.
    activation_scale_policy, weight_scale_policy : str
        'float' (default), 'pot_nearest' or 'pot_ceil'; weight also allows 'pot_mse'.
    adaround : AdaroundConfig or None
        None means RTN only; a configuration enables subsequent fixed-encoding AdaRound.
    weight_group_size : None
        Must remain None; per-group weights are unsupported.

    scheme() returns the engine's policy dict, not a public Scheme object.

    Examples
    --------
    >>> from opennpu_quant import QuantizationConfig, CalibrationConfig
    >>> config = QuantizationConfig(scope='all', weight_granularity='per_channel',
    ...     calibration=CalibrationConfig.percentile(99.99))
    >>> assert config.representation == 'QDQ'
    """

    activation_precision: QuantPrecision = field(default_factory=QuantPrecision)
    weight_precision: QuantPrecision = field(default_factory=QuantPrecision)
    activation_symmetric: bool = True
    weight_symmetric: bool = True
    activation_granularity: str = "per_tensor"
    weight_granularity: str = "per_tensor"
    scope: str = "basic"
    representation: str = "QDQ"
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    activation_scale_policy: str = "float"
    weight_scale_policy: str = "float"
    adaround: AdaroundConfig | None = None
    weight_group_size: int | None = None

    def __post_init__(self):
        scale_policies(vars(self))
        if self.adaround is not None and not isinstance(self.adaround, AdaroundConfig):
            raise TypeError("adaround must be AdaroundConfig or None")
        if (
            self.activation_precision != QuantPrecision()
            or self.weight_precision != QuantPrecision()
        ):
            raise ValueError("Only signed INT8 activation and weight are implemented")
        if not self.weight_symmetric or self.activation_granularity != "per_tensor":
            raise ValueError(
                "Only symmetric weight / per-tensor activation are implemented"
            )
        if self.weight_granularity not in ("per_tensor", "per_channel"):
            raise ValueError("Unsupported weight granularity")
        if self.weight_group_size is not None:
            raise ValueError("weight_group_size is only valid for per_group weights")
        if self.scope not in ("basic", "all") or self.representation != "QDQ":
            raise ValueError("Only basic/all ONNX QDQ are implemented")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        values = dict(value)
        for name in ("activation_precision", "weight_precision"):
            if name in values:
                values[name] = QuantPrecision(**values[name])
        if values.get("adaround") is not None:
            values["adaround"] = AdaroundConfig.from_dict(values["adaround"])
        if "calibration" in values:
            values["calibration"] = CalibrationConfig.from_dict(values["calibration"])
        return cls(**values)

    def scheme(self):
        result = dict(
            scope=self.scope,
            activation_symmetric=self.activation_symmetric,
            weight_per_channel=self.weight_granularity == "per_channel",
            method=self.calibration.method,
        )
        if self.calibration.percentile_value is not None:
            result["percentile"] = self.calibration.percentile_value
        if (
            self.activation_scale_policy != "float"
            or self.weight_scale_policy != "float"
        ):
            result.update(
                activation_scale_policy=self.activation_scale_policy,
                weight_scale_policy=self.weight_scale_policy,
            )
        if self.adaround is not None:
            result["adaround"] = self.adaround.to_dict()
        return result

    @property
    def name(self):
        method = self.calibration.method.lower()
        if self.calibration.percentile_value is not None:
            method += "_" + str(self.calibration.percentile_value).replace(".", "_")
        suffix = (
            ""
            if (self.activation_scale_policy, self.weight_scale_policy)
            == ("float", "float")
            else f"__as_{self.activation_scale_policy}__ws_{self.weight_scale_policy}"
        )
        if self.adaround is not None:
            suffix += "__adaround_" + self.adaround.identity[:16]
        granularity = {
            "per_tensor": "_w_pt_",
            "per_channel": "_w_pc_",
        }[self.weight_granularity]
        return (
            ("a_sym" if self.activation_symmetric else "a_asym")
            + granularity
            + method
            + suffix
        )
