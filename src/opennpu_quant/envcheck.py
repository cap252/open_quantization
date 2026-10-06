from importlib.metadata import version, PackageNotFoundError
import platform

REFERENCE = {
    "numpy": "1.26.4",
    "onnx": "1.17.0",
    "onnxruntime-gpu": "1.20.2",
    "torch": "2.5.1",
    "torchvision": "0.20.1",
    "pillow": "11.1.0",
    "opencv-python-headless": "4.11.0.86",
    "pycocotools": "2.0.8",
    "tensorflow-cpu": "2.15.1",
    "tf2onnx": "1.16.1",
}


def info():
    packages = {}
    for name, reference in REFERENCE.items():
        try:
            installed = version(name)
        except PackageNotFoundError:
            installed = None
        packages[name] = dict(
            installed=installed,
            reference=reference,
            matches=installed is not None and installed.split("+")[0] == reference,
        )
    try:
        import onnxruntime as ort

        providers = ort.get_available_providers()
    except ImportError:
        providers = []
    return dict(
        python=platform.python_version(),
        platform=platform.platform(),
        packages=packages,
        providers=providers,
        supported_platform="Linux x86_64",
        full_validation_verified=False,
    )
