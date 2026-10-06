def _utils():
    from .session import runtime

    runtime()
    from onnxruntime.quantization import quant_utils

    return quant_utils


def compute_scale_zp(*args, **kwargs):
    return _utils().compute_scale_zp(*args, **kwargs)


def get_qmin_qmax_for_qType(*args, **kwargs):
    return _utils().get_qmin_qmax_for_qType(*args, **kwargs)


def quantize_data(*args, **kwargs):
    return _utils().quantize_data(*args, **kwargs)


def quantize_nparray(*args, **kwargs):
    return _utils().quantize_nparray(*args, **kwargs)


def adjust_weight_scale(*args, **kwargs):
    from .session import runtime

    runtime()
    from onnxruntime.quantization.qdq_quantizer import QDQQuantizer

    return QDQQuantizer._adjust_weight_scale_for_int32_bias(None, *args, **kwargs)


def histogram_collector(*args, **kwargs):
    from .session import runtime

    runtime()
    from onnxruntime.quantization.calibrate import HistogramCollector

    return HistogramCollector(*args, **kwargs)


def compute_data_quant_params(*args, **kwargs):
    return _utils().compute_data_quant_params(*args, **kwargs)
