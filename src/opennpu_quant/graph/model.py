import copy
import hashlib
from pathlib import Path
import onnx
from onnx.external_data_helper import _get_all_tensors, convert_model_from_external_data
from opennpu_quant._io import object_hash, sha256


def load_model(source):
    if isinstance(source, (str, Path)):
        path = Path(source).resolve()
        header = onnx.load(path, load_external_data=False)
        for tensor in _get_all_tensors(header):
            if tensor.data_location == onnx.TensorProto.EXTERNAL:
                location = dict(((v.key, v.value) for v in tensor.external_data)).get(
                    "location", ""
                )
                if not location or not (
                    path.parent / location
                ).resolve().is_relative_to(path.parent):
                    raise ValueError("External data must stay within the model bundle")
        model = onnx.load(path, load_external_data=True)
    elif isinstance(source, onnx.ModelProto):
        model = copy.deepcopy(source)
        for tensor in _get_all_tensors(model):
            if tensor.data_location == onnx.TensorProto.EXTERNAL and (
                not tensor.HasField("raw_data")
            ):
                raise ValueError(
                    "Unresolved ONNX external data; provide its original bundle path"
                )
    else:
        raise TypeError("Expected ONNX path or ModelProto")
    convert_model_from_external_data(model)
    for tensor in _get_all_tensors(model):
        tensor.ClearField("data_location")
    return model


def model_identity(source):
    return hashlib.sha256(
        load_model(source).SerializeToString(deterministic=True)
    ).hexdigest()


def bundle_identity(path):
    path = Path(path).resolve()
    model = onnx.load(path, load_external_data=False)
    rows = {path.name: sha256(path)}
    for tensor in _get_all_tensors(model):
        if tensor.data_location == onnx.TensorProto.EXTERNAL:
            name = dict(((v.key, v.value) for v in tensor.external_data))["location"]
            child = (path.parent / name).resolve()
            if not child.is_relative_to(path.parent):
                raise ValueError("External data escapes bundle")
            rows[name] = sha256(child)
    return {"identity": object_hash(rows), "files": rows}


def save_model(model, path):
    """Save a checked model through a temporary file and return bundle hashes.

    Parameters
    ----------
    model : path-like or onnx.ModelProto
        Copied before saving. External weights are materialized; an unresolved
        external-data ModelProto must instead be supplied as its bundle path.
    path : path-like
        Destination ONNX file; parent directories are created. An existing
        destination is replaced only after writing its .pending file.

    Returns
    -------
    dict
        {"identity": bundle_hash, "files": {filename: sha256}}; not a hash
        string. The bundle identity includes filenames and file contents.

    Examples
    --------
    With a prepared model:

    >>> from tempfile import TemporaryDirectory
    >>> from opennpu_quant import save_model
    >>> with TemporaryDirectory() as folder:
    ...     saved = save_model(model, folder + '/model.onnx')
    ...     assert 'model.onnx' in saved['files']
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    owned = load_model(model)
    onnx.checker.check_model(owned)
    temporary = path.with_suffix(path.suffix + ".pending")
    onnx.save_model(owned, temporary)
    temporary.replace(path)
    return bundle_identity(path)
