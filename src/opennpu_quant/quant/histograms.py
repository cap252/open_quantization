from pathlib import Path
import numpy as np
from opennpu_quant._io import atomic_json, read_json, sha256


class FixedHistograms:
    """Track signed and absolute histograms in fixed, recorded FP32 ranges."""

    def __init__(self, ranges, bins=2048):
        self.bins = bins
        self.rows = {}
        for name, limits in sorted(ranges.items()):
            low = np.float32(np.asarray(limits["lowest"]).item())
            high = np.float32(np.asarray(limits["highest"]).item())
            if not np.isfinite([low, high]).all() or low > high:
                raise ValueError("Invalid MinMax range: " + name)
            # Fixed numerical headroom, chosen before evaluating candidate accuracy.
            bound = np.float32(max(abs(low), abs(high)) * 1.00001 + 1e-6)
            self.rows[name] = dict(
                low=low,
                high=high,
                bound=bound,
                signed_edges=np.linspace(-bound, bound, bins + 1, dtype=np.float32),
                absolute_edges=np.linspace(0, bound, bins + 1, dtype=np.float32),
                signed=np.zeros(bins, np.int64),
                absolute=np.zeros(bins, np.int64),
                count=0,
                images=0,
            )

    def add(self, name, values):
        row = self.rows[name]
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        if not values.size or not np.isfinite(values).all():
            raise ValueError("Empty/non-finite calibration tensor: " + name)
        signed, signed_edges = np.histogram(
            values, bins=self.bins, range=(-row["bound"], row["bound"])
        )
        absolute, absolute_edges = np.histogram(
            np.abs(values), bins=self.bins, range=(np.float32(0), row["bound"])
        )
        if not np.array_equal(signed_edges, row["signed_edges"]) or not np.array_equal(
            absolute_edges, row["absolute_edges"]
        ):
            raise ValueError("Uniform-bin fast path changed recorded histogram edges")
        if signed.sum() != values.size or absolute.sum() != values.size:
            raise ValueError("Histogram range missed tensor values: " + name)
        row["signed"] += signed
        row["absolute"] += absolute
        row["count"] += values.size
        row["images"] += 1

    @classmethod
    def load(cls, path):
        meta = read_json(Path(path).with_suffix(".json"))
        if sha256(path) != meta["histogram_sha256"]:
            raise ValueError("Histogram cache changed")
        result = cls(
            {
                r["name"]: dict(lowest=r["low"], highest=r["high"])
                for r in meta["tensors"]
            },
            meta["bins"],
        )
        with np.load(path, allow_pickle=False) as arrays:
            for index, item in enumerate(meta["tensors"]):
                row = result.rows[item["name"]]
                for key in ("signed", "absolute", "signed_edges", "absolute_edges"):
                    row[key] = arrays[f"{index}_{key}"].copy()
                row["count"], row["images"] = item["count"], item["images"]
        return result

    def save(self, path):
        arrays = {}
        metadata = []
        for index, (name, row) in enumerate(self.rows.items()):
            for key in ("signed", "absolute", "signed_edges", "absolute_edges"):
                arrays[f"{index}_{key}"] = row[key]
            metadata.append(
                dict(
                    name=name,
                    low=float(row["low"]),
                    high=float(row["high"]),
                    bound=float(row["bound"]),
                    count=int(row["count"]),
                    images=row["images"],
                )
            )
        np.savez_compressed(path, **arrays)
        atomic_json(
            Path(path).with_suffix(".json"),
            dict(
                bins=self.bins,
                tensors=metadata,
                histogram_sha256=sha256(path),
                range_headroom="absmax*1.00001+1e-6; all values counted or fail",
                collection="Fixed edges from independent prior MinMax; one image at a time; signed and absolute histograms",
                threshold_implementation="Installed ORT HistogramCollector.compute_percentile / compute_entropy unchanged",
            ),
        )
