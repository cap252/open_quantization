"""Explicit path ownership: CLI > environment > local YAML > defaults."""

from dataclasses import dataclass
from pathlib import Path
import os
from ._io import read_yaml


@dataclass(frozen=True)
class Paths:
    home: Path
    models: Path
    data: Path
    cache: Path
    runs: Path
    weights: Path
    datasets: dict

    @classmethod
    def resolve(
        cls,
        *,
        home=None,
        models_dir=None,
        data_dir=None,
        cache_dir=None,
        output=None,
        weights_dir=None,
        local=None,
    ):
        path = Path(local or "opennpu_quant.local.yaml")
        if local is not None and not path.exists():
            raise FileNotFoundError(
                f"--local-config file not found: {path}. Pass an existing YAML file or omit --local-config to use defaults."
            )
        config = read_yaml(path) if path.exists() else {}
        if config is None:
            config = {}
        if not isinstance(config, dict):
            raise ValueError(f"Local path settings in {path} must be a mapping")
        allowed = {"home", "models", "data", "cache", "runs", "weights", "datasets"}
        if set(config) - allowed:
            raise ValueError("Unknown local path setting")

        def path_setting(value, key):
            if not isinstance(value, str) or not value.strip() or "\0" in value:
                raise ValueError(
                    f"Local setting {key} in {path} must be a nonempty path string"
                )

        for key in config.keys() - {"datasets"}:
            path_setting(config[key], key)
        datasets = config.get("datasets", {})
        if not isinstance(datasets, dict):
            raise ValueError(f"Local setting datasets in {path} must be a mapping")
        for name, value in datasets.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError(
                    f"Local setting datasets in {path} requires nonempty string names"
                )
            path_setting(value, "datasets." + name)

        def absolute(value):
            return Path(value).expanduser().resolve()

        home = absolute(
            home or os.getenv("OPENNPU_QUANT_HOME") or config.get("home", "workspace")
        )

        def resolve(name, explicit):
            return absolute(
                explicit
                or os.getenv("OPENNPU_QUANT_" + name.upper())
                or config.get(name, home / name)
            )

        return cls(
            home,
            resolve("models", models_dir),
            resolve("data", data_dir),
            resolve("cache", cache_dir),
            resolve("runs", output),
            resolve("weights", weights_dir),
            datasets,
        )

    def to_dict(self):
        return {k: str(v) if isinstance(v, Path) else v for k, v in vars(self).items()}
