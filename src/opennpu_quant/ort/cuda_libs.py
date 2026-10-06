import ctypes
import sysconfig
from pathlib import Path

_handles = []
_loaded = False


def load_cuda_libraries():
    global _loaded
    if _loaded:
        return
    base = Path(sysconfig.get_paths()["purelib"]) / "nvidia"
    # ORT 1.20.2 links the split cuDNN libraries, not just libcudnn.so.
    # No LD_LIBRARY_PATH borrowed from a different environment is needed.
    names = (
        "cuda_runtime/lib/libcudart.so.12",
        "nvjitlink/lib/libnvJitLink.so.12",
        "cublas/lib/libcublasLt.so.12",
        "cublas/lib/libcublas.so.12",
        "cufft/lib/libcufft.so.11",
        "curand/lib/libcurand.so.10",
        "cuda_nvrtc/lib/libnvrtc.so.12",
        "cudnn/lib/libcudnn.so.9",
        "cudnn/lib/libcudnn_graph.so.9",
        "cudnn/lib/libcudnn_ops.so.9",
        "cudnn/lib/libcudnn_adv.so.9",
        "cudnn/lib/libcudnn_cnn.so.9",
        "cudnn/lib/libcudnn_engines_precompiled.so.9",
        "cudnn/lib/libcudnn_engines_runtime_compiled.so.9",
        "cudnn/lib/libcudnn_heuristic.so.9",
    )
    missing = [name for name in names if not (base / name).is_file()]
    if missing:
        raise RuntimeError(
            "CUDA wheel libraries are missing in this interpreter. "
            "Use the CUDA setup profile: " + ", ".join(missing)
        )
    for name in names:
        _handles.append(ctypes.CDLL(str(base / name), mode=ctypes.RTLD_GLOBAL))
    _loaded = True
