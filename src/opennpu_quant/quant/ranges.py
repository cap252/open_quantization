from .config import CalibrationConfig
from ..ort.numerics import histogram_collector


def select_ranges(
    histograms, minmax_ranges, *, config: CalibrationConfig, activation_symmetric
):
    """Select activation ranges without changing the collected statistics."""
    if config.method == "MinMax":
        return minmax_ranges
    if histograms is not None and histograms.bins != config.histogram_bins:
        raise ValueError("Histogram bin count differs from calibration config")
    return _histogram_ranges(
        histograms,
        config.method,
        symmetric=activation_symmetric,
        percentile=config.percentile_value
        if config.percentile_value is not None
        else 99.999,
        quantized_bins=config.quantized_bins,
    )


def _histogram_ranges(histograms, method, *, symmetric, percentile, quantized_bins):
    if method not in ("Percentile", "Entropy"):
        raise ValueError("Expected Percentile or Entropy")
    if histograms is None or not histograms.rows:
        raise ValueError("No histogram samples")
    collector = histogram_collector(
        method=method.lower(),
        symmetric=symmetric,
        num_bins=histograms.bins,
        num_quantized_bins=quantized_bins,
        percentile=percentile,
        scenario="same",
    )
    for name, row in histograms.rows.items():
        if row["images"] <= 0 or row["count"] <= 0:
            raise ValueError("No histogram samples: " + name)
        if method == "Percentile":
            # Symmetric uses |x|; asymmetric trims both signed tails.
            key = "absolute" if symmetric else "signed"
            collector.histogram_dict[name] = (
                row[key],
                row[key + "_edges"],
                row["low"],
                row["high"],
            )
        else:
            # ORT's centered entropy search does not use the symmetric flag.
            collector.histogram_dict[name] = (
                row["signed"],
                row["signed_edges"],
                row["low"],
                row["high"],
                row["bound"],
            )
    return {
        name: dict(lowest=float(limits[0]), highest=float(limits[1]))
        for name, limits in collector.compute_collection_result().items()
    }
