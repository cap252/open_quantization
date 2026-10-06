from collections import OrderedDict
from dataclasses import dataclass


@dataclass(frozen=True)
class ActivationCacheConfig:
    """Byte budgets for AdaRound's intermediate activation cache.

    Attributes
    ----------
    host_bytes : int
        Nonnegative CPU cache ceiling, default 4 GiB; zero disables host retention.
    device_bytes : int
        Nonnegative GPU window ceiling, default 512 MiB; zero keeps windows on host.
    device_reserve_bytes : int
        Free GPU memory to leave available, default 2 GiB.

    Budgets are byte counts, not preallocations. Sampling order and loss arithmetic
    are unchanged by cache capacity. This does not configure the input disk cache.

    Examples
    --------
    >>> from opennpu_quant import ActivationCacheConfig
    >>> cache = ActivationCacheConfig(host_bytes=1024**3, device_bytes=0)
    >>> assert cache.device_bytes == 0
    """

    host_bytes: int = 4 * 1024**3
    device_bytes: int = 512 * 1024**2
    device_reserve_bytes: int = 2 * 1024**3

    def __post_init__(self):
        for value in (self.host_bytes, self.device_bytes, self.device_reserve_bytes):
            if type(value) is not int or value < 0:
                raise ValueError(
                    "Cache budgets must be nonnegative integer byte counts"
                )


class LayerActivationCache:
    def __init__(self, config):
        self.config = config
        self.entries = OrderedDict()
        self.bytes = 0
        self.metrics = dict(
            hits=0,
            misses=0,
            evictions=0,
            host_peak_bytes=0,
            device_windows=0,
            device_fallback_windows=0,
            device_upload_bytes=0,
        )

    def get(self, key):
        value = self.entries.get(key)
        if value is None:
            self.metrics["misses"] += 1
        else:
            self.entries.move_to_end(key)
            self.metrics["hits"] += 1
        return value

    def put(self, key, pair):
        size = sum(a.nbytes for a in pair)
        if size > self.config.host_bytes:
            return
        while self.entries and self.bytes + size > self.config.host_bytes:
            _, old = self.entries.popitem(last=False)
            self.bytes -= sum(a.nbytes for a in old)
            self.metrics["evictions"] += 1
        self.entries[key] = pair
        self.bytes += size
        self.metrics["host_peak_bytes"] = max(
            self.metrics["host_peak_bytes"], self.bytes
        )

    def device_window(self, items, device):
        import torch

        size = sum(x.nbytes + y.nbytes for x, y in items)
        if (
            device != "cuda"
            or not self.config.device_bytes
            or size > self.config.device_bytes
            or size + self.config.device_reserve_bytes > torch.cuda.mem_get_info()[0]
        ):
            self.metrics["device_fallback_windows"] += 1
            return items
        converted = []
        try:
            for x, y in items:
                converted.append(
                    (torch.tensor(x, device=device), torch.tensor(y, device=device))
                )
        except torch.cuda.OutOfMemoryError:
            converted.clear()
            torch.cuda.empty_cache()
            self.metrics["device_fallback_windows"] += 1
            return items
        self.metrics["device_windows"] += 1
        self.metrics["device_upload_bytes"] += size
        return converted
