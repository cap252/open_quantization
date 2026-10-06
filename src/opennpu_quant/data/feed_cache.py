from pathlib import Path
import hashlib
import os
import shutil
import tempfile
import numpy as np
from .._io import atomic_json, object_hash, read_json, sha256
from .._locking import RunLock
from .._feeds import feed_digest, _VerifiedFeed, _fingerprint


class FeedCache:
    def __init__(self, directory, *, identity, verify_source):
        self.directory = Path(directory)
        self.reused = False
        self.verify_source = verify_source
        self.manifest = read_json(self.directory / "manifest.json")
        if (
            self.manifest.get("schema_version") != 1
            or self.manifest.get("state") != "completed"
        ):
            raise ValueError("Incomplete feed cache")
        if self.manifest["identity"] != object_hash(identity):
            raise ValueError("Feed cache preprocessing/source identity differs")
        count = self.manifest["sample_count"]
        entries = self.manifest["tensors"]
        if (
            type(count) is not int
            or count < 1
            or not entries
            or len({v["name"] for v in entries}) != len(entries)
            or len({v["file"] for v in entries}) != len(entries)
        ):
            raise ValueError("Invalid feed cache tensor/count manifest")
        for entry in entries:
            if (
                entry["dtype"] != "float32"
                or not entry["shape"]
                or entry["shape"][0] != 1
                or any(type(n) is not int or n < 1 for n in entry["shape"])
            ):
                raise ValueError("Invalid cached FP32 batch-one tensor")
        signatures = self.manifest["feed_digests"]
        if (
            count < 1
            or len(signatures) != count
            or hashlib.sha256("".join(signatures).encode()).hexdigest()
            != self.manifest["samples_identity"]
        ):
            raise ValueError("Feed cache sample identity is inconsistent")
        fingerprints = self.verify()
        arrays = {
            entry["name"]: np.load(
                self.directory / entry["file"], mmap_mode="r", allow_pickle=False
            )
            for entry in self.manifest["tensors"]
        }
        for entry in self.manifest["tensors"]:
            value = arrays[entry["name"]]
            if (
                list(value.shape) != [count, *entry["shape"]]
                or str(value.dtype) != entry["dtype"]
            ):
                raise ValueError("Feed cache tensor layout differs")
        for index, expected in enumerate(signatures):
            if (
                feed_digest({name: array[index] for name, array in arrays.items()})
                != expected
            ):
                raise ValueError("Feed cache stored sample digest differs from payload")
        if any(_fingerprint(path) != fingerprint for path, fingerprint in fingerprints):
            raise ValueError("Feed cache changed while opening")

    def verify(self):
        self.verify_source()
        fingerprints = []
        for entry in self.manifest["tensors"]:
            path = (self.directory / entry["file"]).resolve()
            if not path.is_relative_to(self.directory.resolve()):
                raise ValueError("Feed cache path escaped cache directory")
            before = _fingerprint(path)
            if (
                before[2] != entry["bytes"]
                or sha256(path) != entry["sha256"]
                or _fingerprint(path) != before
            ):
                raise ValueError("Feed cache payload changed: " + entry["file"])
            fingerprints.append((path, before))
        return tuple(fingerprints)

    def __call__(self):
        # Recheck original bytes and packed payload on every replay; never silently
        # return stale inputs when a dataset or cache file changes.
        files = self.verify()
        arrays = {}
        try:
            for entry in self.manifest["tensors"]:
                array = np.load(
                    self.directory / entry["file"], mmap_mode="r", allow_pickle=False
                )
                if (
                    list(array.shape)
                    != [self.manifest["sample_count"], *entry["shape"]]
                    or str(array.dtype) != entry["dtype"]
                ):
                    raise ValueError("Feed cache tensor layout differs")
                arrays[entry["name"]] = array
            for index, signature in enumerate(self.manifest["feed_digests"]):
                feed = _VerifiedFeed(
                    {name: array[index] for name, array in arrays.items()},
                    signature,
                    files,
                )
                feed.verified_digest()
                yield feed
            self.verify_source()
            if any(_fingerprint(path) != fingerprint for path, fingerprint in files):
                raise ValueError("Cached payload changed during replay")
        finally:
            # Caller arrays remain usable for the current yield; np.memmap owns
            # its mapping through all slices. Do not force-close active slices.
            arrays.clear()


def prepare_feed_cache(
    directory,
    factory,
    *,
    identity,
    verify_source,
    expected_samples,
    max_bytes,
    reserve=None,
):
    """Create or validate a replayable disk cache of preprocessed FP32 feeds.

    Parameters
    ----------
    directory : path-like
        Cache folder. An existing cache must pass content and source validation.
    factory : callable
        Zero-argument fresh iterator of input-name-to-array dicts; fixed shapes,
        finite FP32 values, batch size one, stable keys, count, order and values.
    identity : JSON-serializable object
        Caller-owned evidence covering source data, preprocessing, settings and
        dependencies that affect feeds. Matching names alone do not suffice.
    verify_source : callable
        Zero-argument check that raises if the original source has changed.
        Called during preparation and replay; do not replace it with a no-op.
    expected_samples : int
        Positive exact number of feeds.
    max_bytes : int
        Positive disk budget in bytes, including storage overhead.
    reserve : callable, optional
        Called with estimated bytes before writing a new cache; may raise.

    Returns
    -------
    FeedCache
        Callable fresh-feed factory with read-only arrays, manifest and reused
        flag. Invalid identity/content raises; insufficient space/budget raises
        MemoryError. This low-level API does not infer preprocessing provenance.

    Examples
    --------
    Assume feeds yields four samples, identity describes their inputs, and
    verify_source verifies the original inputs on each call:

    >>> from tempfile import TemporaryDirectory
    >>> from opennpu_quant import prepare_feed_cache
    >>> with TemporaryDirectory() as folder:
    ...     cached = prepare_feed_cache(folder + '/inputs', feeds, identity=identity,
    ...         verify_source=verify_source, expected_samples=4, max_bytes=1024**2)
    ...     assert len(list(cached())) == 4
    """
    directory = Path(directory)
    if type(expected_samples) is not int or expected_samples < 1:
        raise ValueError("expected_samples must be a positive integer sample count")
    if type(max_bytes) is not int or max_bytes < 1:
        raise ValueError("max_bytes must be a positive integer byte budget")
    if not callable(factory):
        raise ValueError(
            "factory must be a zero-argument callable returning a new iterator of "
            "{input name: array} dicts on every call. Example: factory=lambda: iter(feed_list)"
        )
    if not callable(verify_source):
        raise ValueError(
            "verify_source must be a zero-argument callable that checks source integrity "
            "before each replay and raises if it changed; e.g. verify_source=check_source"
        )
    directory.parent.mkdir(parents=True, exist_ok=True)
    with RunLock(directory.with_name(directory.name + ".lock")):
        if directory.exists():
            cache = FeedCache(directory, identity=identity, verify_source=verify_source)
            if cache.manifest["sample_count"] != expected_samples:
                raise ValueError("Reused calibration sample count changed")
            if sum(entry["bytes"] for entry in cache.manifest["tensors"]) > max_bytes:
                raise MemoryError("Existing calibration cache exceeds its disk budget")
            cache.reused = True
            return cache
        verify_source()
        temporary = Path(
            tempfile.mkdtemp(prefix=directory.name + ".building-", dir=directory.parent)
        )
        arrays = {}
        try:
            iterator = iter(factory())
            first = next(iterator, None)
            if not first:
                raise ValueError("Empty calibration feed factory")
            tensors = []
            for index, (name, value) in enumerate(first.items()):
                value = np.asarray(value)
                if (
                    value.dtype != np.float32
                    or value.ndim < 1
                    or value.shape[0] != 1
                    or not value.size
                    or not np.isfinite(value).all()
                ):
                    raise ValueError("Cache requires finite FP32 batch-one inputs")
                tensors.append(
                    dict(
                        name=name,
                        shape=list(value.shape),
                        dtype=str(value.dtype),
                        file=f"tensor_{index}.npy",
                    )
                )
            estimated = sum(
                np.prod(t["shape"], dtype=np.int64) * 4 * expected_samples + 4096
                for t in tensors
            )
            if estimated > max_bytes:
                raise MemoryError("Calibration cache exceeds its disk budget")
            if shutil.disk_usage(temporary).free < estimated:
                raise MemoryError("Insufficient free space for calibration input cache")
            if reserve is not None:
                reserve(int(estimated))
            for entry in tensors:
                arrays[entry["name"]] = np.lib.format.open_memmap(
                    temporary / entry["file"],
                    mode="w+",
                    dtype=np.float32,
                    shape=(expected_samples, *entry["shape"]),
                )
            signatures = []
            from itertools import chain

            for index, feed in enumerate(chain([first], iterator)):
                if index >= expected_samples or set(feed) != set(arrays):
                    raise ValueError("Calibration sample count or input names changed")
                for entry in tensors:
                    value = np.asarray(feed[entry["name"]])
                    if (
                        list(value.shape) != entry["shape"]
                        or str(value.dtype) != entry["dtype"]
                        or not np.isfinite(value).all()
                    ):
                        raise ValueError("Variable/nonfinite calibration tensor layout")
                    arrays[entry["name"]][index] = value
                signatures.append(feed_digest(feed))
            if len(signatures) != expected_samples:
                raise ValueError("Calibration sample count changed")
            for array in arrays.values():
                array.flush()
            arrays.clear()
            for entry in tensors:
                path = temporary / entry["file"]
                entry.update(bytes=path.stat().st_size, sha256=sha256(path))
                path.chmod(0o444)
            verify_source()
            record = dict(
                schema_version=1,
                state="completed",
                identity=object_hash(identity),
                source_identity=identity,
                sample_count=expected_samples,
                tensors=tensors,
                feed_digests=signatures,
                samples_identity=hashlib.sha256(
                    "".join(signatures).encode()
                ).hexdigest(),
            )
            atomic_json(temporary / "manifest.json", record)
            os.replace(temporary, directory)
            return FeedCache(directory, identity=identity, verify_source=verify_source)
        finally:
            arrays.clear()
            if temporary.exists():
                shutil.rmtree(temporary)
