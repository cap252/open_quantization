from pathlib import Path
import shutil, tarfile, zipfile, urllib.request, tempfile
from ._io import read_yaml, sha256
from ._locking import RunLock


def download(url, destination, *, sha256_expected):
    if not isinstance(sha256_expected, str) or len(sha256_expected) != 64:
        raise ValueError("Expected SHA256 is required")
    destination = Path(destination)
    with RunLock(destination.with_suffix(destination.suffix + ".lock")):
        if destination.exists():
            if sha256(destination) != sha256_expected:
                raise ValueError(
                    "Existing download checksum mismatch: " + destination.name
                )
            return destination
        temporary = destination.with_suffix(destination.suffix + ".pending")
        try:
            with (
                urllib.request.urlopen(url, timeout=60) as response,
                temporary.open("wb") as stream,
            ):
                shutil.copyfileobj(response, stream, length=1024 * 1024)
            if sha256(temporary) != sha256_expected:
                raise ValueError("Download checksum mismatch: " + destination.name)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
    return destination


def extract_archive(archive, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)

    def checked(name):
        result = (destination / name).resolve()
        if not result.is_relative_to(destination.resolve()):
            raise ValueError("Archive path escapes destination")
        return result

    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            for member in z.infolist():
                checked(member.filename)
                if (member.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError("Archive links are not supported")
            z.extractall(destination)
    else:
        with tarfile.open(archive) as t:
            for member in t.getmembers():
                checked(member.name)
                if not (member.isdir() or member.isfile()):
                    raise ValueError("Archive links/special files are not supported")
            t.extractall(destination, filter="data")


def fetch_assets(kind, manifest_path, models, paths):
    manifest = read_yaml(Path(manifest_path))
    results = []
    for name in models:
        if "/" in name or ".." in name:
            raise ValueError("Unsafe model name")
        row = manifest[kind][name]
        archive = download(
            row["url"],
            paths.cache
            / "downloads"
            / (kind + "_" + name + "_" + row["sha256"] + ".tar.gz"),
            sha256_expected=row["sha256"],
        )
        destination = (
            paths.models / name
            if kind == "models"
            else paths.cache / "calibration" / name / "reference"
        )
        with RunLock(destination.parent / (destination.name + ".fetch.lock")):
            if destination.exists():
                raise ValueError(
                    "Asset destination already exists; verify or use a new workspace: "
                    + str(destination)
                )
            with tempfile.TemporaryDirectory(
                dir=destination.parent, prefix=".asset_"
            ) as temp:
                unpacked = Path(temp) / "unpacked"
                extract_archive(archive, unpacked)
                unpacked.replace(destination)
        results.append(str(destination))
    return results
