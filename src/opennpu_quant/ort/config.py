from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class OrtConfig:
    """Immutable ONNX Runtime execution settings; constructing it runs no model.

    Attributes
    ----------
    providers, provider_options : tuple
        Ordered CPU/CUDA providers and matching option tuples. Use cpu() or
        cuda(device=0, memory_limit=6 * 1024**3); CUDA memory is in bytes.
    intra_op_threads, inter_op_threads : int
        Nonnegative thread counts; both default to 1.
    optimization : str
        Default ORT_DISABLE_ALL, required for calibration and quantization.
    allowed_versions, strict_versions : tuple, bool
        Default ('1.20.2',) and False; unverified versions warn, or fail in
        strict mode. Hardware availability is checked when a session is made.
    scratch : path-like or None
        Parent directory for temporary files; None uses the system default.
    profile_samples : int
        0 or 1 (default); CUDA evaluation requires 1 for placement checks.

    Examples
    --------
    >>> from opennpu_quant import OrtConfig
    >>> cpu = OrtConfig.cpu()
    >>> gpu = OrtConfig.cuda(0, memory_limit=6 * 1024**3)
    >>> assert cpu.optimization == gpu.optimization == 'ORT_DISABLE_ALL'
    """

    providers: tuple[str, ...] = ("CPUExecutionProvider",)
    provider_options: tuple[tuple[tuple[str, str], ...], ...] = ()
    intra_op_threads: int = 1
    inter_op_threads: int = 1
    optimization: str = "ORT_DISABLE_ALL"
    allowed_versions: tuple[str, ...] = ("1.20.2",)
    strict_versions: bool = False
    scratch: str | None = None
    profile_samples: int = 1

    def __post_init__(self):
        object.__setattr__(self, "providers", tuple(self.providers))
        object.__setattr__(self, "allowed_versions", tuple(self.allowed_versions))
        object.__setattr__(
            self,
            "provider_options",
            tuple((tuple((tuple(p) for p in row)) for row in self.provider_options)),
        )
        if not self.providers or len(set(self.providers)) != len(self.providers):
            raise ValueError("Ordered unique providers required")
        if any(
            (
                p not in ("CPUExecutionProvider", "CUDAExecutionProvider")
                for p in self.providers
            )
        ):
            raise ValueError("Only CPU and CUDA providers are supported")
        if self.provider_options and len(self.providers) != len(self.provider_options):
            raise ValueError("provider_options must correspond to providers")
        if min(self.intra_op_threads, self.inter_op_threads) < 0:
            raise ValueError("Threads must be nonnegative")
        if type(self.profile_samples) is not int or self.profile_samples not in (0, 1):
            raise ValueError(
                "The bounded execution profile supports zero or one sample"
            )
        if "CUDAExecutionProvider" in self.providers and self.profile_samples != 1:
            raise ValueError("CUDA evaluation requires one profiled sample")
        if self.optimization not in (
            "ORT_DISABLE_ALL",
            "ORT_ENABLE_BASIC",
            "ORT_ENABLE_EXTENDED",
            "ORT_ENABLE_ALL",
        ):
            raise ValueError("Unknown optimization policy")

    @classmethod
    def cpu(cls, **kwargs):
        return cls(**kwargs)

    @classmethod
    def cuda(cls, device=0, *, memory_limit=6 * 1024**3, **kwargs):
        if type(device) is not int or device < 0 or memory_limit <= 0:
            raise ValueError(
                "CUDA device and memory limit must be nonnegative/positive"
            )
        options = {
            "device_id": str(device),
            "use_tf32": "0",
            "gpu_mem_limit": str(memory_limit),
            "cudnn_conv_algo_search": "HEURISTIC",
            "cudnn_conv_use_max_workspace": "0",
        }
        return cls(
            providers=("CUDAExecutionProvider", "CPUExecutionProvider"),
            provider_options=(tuple(options.items()), ()),
            **kwargs,
        )

    def to_dict(self):
        return asdict(self)
