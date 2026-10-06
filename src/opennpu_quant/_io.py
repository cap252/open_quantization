import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_yaml(path):
    """Load user YAML and retain its file and parser location in input errors."""
    import yaml

    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        mark = getattr(error, "problem_mark", None)
        location = (
            f"line {mark.line + 1}, column {mark.column + 1}"
            if mark
            else "unknown location"
        )
        problem = getattr(error, "problem", None) or str(error)
        raise ValueError(
            f"Invalid YAML in {path} ({location}): {problem}. Check YAML syntax and indentation."
        ) from None


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".pending")

    def finite_json(v):
        if isinstance(v, float) and (not math.isfinite(v)):
            return {"nonfinite_float": str(v)}
        if isinstance(v, dict):
            return {k: finite_json(item) for k, item in v.items()}
        if isinstance(v, (list, tuple)):
            return [finite_json(item) for item in v]
        return v

    temporary.write_text(
        json.dumps(finite_json(value), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def object_hash(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode()
    ).hexdigest()
