from pathlib import Path
from dataclasses import fields
from .._io import atomic_json, read_json, sha256
from .._locking import RunLock
from .calibration import load_statistics, validate_statistics


def save_statistics(stats, directory):
    """Publish the completion manifest after writing statistics and hashes."""
    directory = Path(directory)
    with RunLock(directory / ".lock"):
        generation = directory / "generations" / stats.statistics_identity
        generation.mkdir(parents=True, exist_ok=True)
        stats.histograms.save(generation / "histograms.npz")
        atomic_json(
            generation / "statistics.json",
            {
                f.name: getattr(stats, f.name)
                for f in fields(stats)
                if f.name != "histograms"
            },
        )
        hashes = {
            n: sha256(generation / n)
            for n in ("histograms.npz", "histograms.json", "statistics.json")
        }
        atomic_json(
            directory / "complete.json",
            dict(generation=str(generation.relative_to(directory)), files=hashes),
        )


def read_statistics(directory, model, *, config, ort, samples_identity=None):
    directory = Path(directory)
    with RunLock(directory / ".lock"):
        proof = read_json(directory / "complete.json")
        generation = (directory / proof["generation"]).resolve()
        if not generation.is_relative_to(directory.resolve()):
            raise ValueError("Unsafe statistics path")
        if set(proof["files"]) != {
            "histograms.npz",
            "histograms.json",
            "statistics.json",
        }:
            raise ValueError("Incomplete statistics manifest")
        if any(
            sha256(generation / n) != digest for n, digest in proof["files"].items()
        ):
            raise ValueError("Statistics payload was modified")
        stats = load_statistics(generation)
        validate_statistics(
            model, stats, config=config, ort=ort, samples_identity=samples_identity
        )
        return stats
