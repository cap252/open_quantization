#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
device=cpu
with_data=0
while (($#)); do
  case "$1" in
    --device) device="$2"; shift 2 ;;
    --data) with_data=1; shift ;;
    --help|-h) echo 'Usage: ./setup.sh [--device cpu|cuda] [--data]'; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ $(uname -s) == Linux && $(uname -m) == x86_64 ]] || { echo 'First supported platform: Linux x86_64' >&2; exit 2; }
[[ "$device" == cpu || "$device" == cuda ]] || { echo 'device must be cpu or cuda' >&2; exit 2; }
mkdir -p "$root/.tools"
if command -v uv >/dev/null 2>&1; then
  uv_bin=$(command -v uv)
else
  uv_bin="$root/.tools/uv"
  if [[ ! -x "$uv_bin" ]]; then
    base='https://github.com/astral-sh/uv/releases/download/0.8.22'
    archive='uv-x86_64-unknown-linux-gnu.tar.gz'
    temp=$(mktemp -d "$root/.tools/bootstrap.XXXXXX")
    trap 'rm -rf -- "$temp"' EXIT
    curl --fail --location "$base/$archive" -o "$temp/$archive"
    curl --fail --location "$base/$archive.sha256" -o "$temp/checksum"
    expected=$(awk '{print $1}' "$temp/checksum")
    actual=$(sha256sum "$temp/$archive" | awk '{print $1}')
    [[ "$actual" == "$expected" ]] || { echo 'uv checksum mismatch' >&2; exit 1; }
    tar -xzf "$temp/$archive" -C "$temp"
    cp "$temp/uv-x86_64-unknown-linux-gnu/uv" "$uv_bin"
    chmod +x "$uv_bin"
  fi
fi
export UV_PYTHON_INSTALL_DIR="$root/.tools/python"
export UV_CACHE_DIR="$root/.tools/cache"
"$uv_bin" python install 3.11.13
if [[ ! -d "$root/.venv" ]]; then "$uv_bin" venv --python 3.11.13 "$root/.venv"; fi
"$root/.venv/bin/python" -c 'import sys; assert sys.version_info[:3] == (3, 11, 13), "Use a new workspace for a different Python version"'
profile=cpu
[[ "$device" == cpu ]] || profile=cuda124
"$uv_bin" pip sync --python "$root/.venv/bin/python" --index-strategy unsafe-best-match "$root/requirements/lock-linux-py311-$profile.txt"
"$uv_bin" pip install --python "$root/.venv/bin/python" --no-deps -e "$root"
if [[ "$with_data" == 1 ]]; then
  "$uv_bin" pip install --python "$root/.venv/bin/python" -r "$root/requirements/data-linux-py311-$profile.txt"
fi
"$uv_bin" pip check --python "$root/.venv/bin/python"
echo "Ready: source $root/.venv/bin/activate"
echo 'Then: opennpu-quant demo --adaround'
