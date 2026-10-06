__version__ = "0.1.0"
_EXPORTS = {
    "OrtConfig": ("ort.config", "OrtConfig"),
    "Scheme": ("quant.scheme", "Scheme"),
    "CalibrationConfig": ("quant.config", "CalibrationConfig"),
    "QuantizationConfig": ("quant.config", "QuantizationConfig"),
    "AdaroundConfig": ("quant.config", "AdaroundConfig"),
    "ActivationCacheConfig": ("quant.adaround_cache", "ActivationCacheConfig"),
    "CalibrationStats": ("quant.calibration", "CalibrationStats"),
    "QuantizationResult": ("quant.api", "QuantizationResult"),
    "prepare_feed_cache": ("data.feed_cache", "prepare_feed_cache"),
    "calibrate": ("api", "calibrate"),
    "quantize": ("api", "quantize"),
    "apply_adaround": ("api", "apply_adaround"),
    "evaluate": ("evaluation.loop", "evaluate"),
    "save_model": ("graph.model", "save_model"),
    "EvaluationSample": ("models.contracts", "EvaluationSample"),
    "TopKClassification": ("evaluation.metrics", "TopKClassification"),
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    from importlib import import_module

    module, symbol = _EXPORTS[name]
    value = getattr(import_module("." + module, __name__), symbol)
    globals()[name] = value
    return value
