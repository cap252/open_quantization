import hashlib
import numpy as np
from ._io import object_hash


def feed_digest(feeds):
    if isinstance(feeds, _VerifiedFeed):
        return feeds.verified_digest()
    digest = hashlib.sha256()
    for name in sorted(feeds):
        value = np.asarray(feeds[name])
        if value.dtype.hasobject:
            raise ValueError("Object arrays are not supported")
        digest.update(object_hash([name, str(value.dtype), list(value.shape)]).encode())
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def _fingerprint(path):
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


class _VerifiedFeed(dict):
    """Digest of bytes verified from an unchanged, read-only mmap backing file."""

    def __init__(self, values, signature, files):
        super().__init__(values)
        self._signature = signature
        self._values = tuple(
            (name, value, value.dtype.str, value.shape, value.strides)
            for name, value in values.items()
        )
        self._files = files

    def verified_digest(self):
        if len(self) != len(self._values):
            raise ValueError("Cached feed keys changed")
        for name, value, dtype, shape, strides in self._values:
            if (
                self.get(name) is not value
                or value.flags.writeable
                or value.dtype.str != dtype
                or value.shape != shape
                or value.strides != strides
            ):
                raise ValueError("Cached read-only feed was replaced or altered")
        if any(_fingerprint(path) != fingerprint for path, fingerprint in self._files):
            raise ValueError("Cached payload changed after verification")
        return self._signature

    def _immutable(self, *args, **kwargs):
        raise TypeError("Verified cached feeds are read-only")

    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = (
        __ior__
    ) = _immutable
