import numpy as np


def run_tensors(session, names, feed):
    """Preserve requested order and independent passthrough input copies.

    ORT 1.20.2 CUDA input-as-output fetches can race the input H2D copy.
    Actual node outputs still execute through ORT. Only names declared as
    graph inputs are read from caller feeds; arbitrary extra feed keys cannot
    replace computed tensors. Pure passthrough requests need no session run.
    """
    inputs = {value.name: feed[value.name] for value in session.get_inputs()}
    computed = list(dict.fromkeys(name for name in names if name not in inputs))
    passthrough = {
        name: np.array(inputs[name], copy=True) for name in names if name in inputs
    }
    values = session.run(computed, inputs) if computed else []
    if len(values) != len(computed):
        raise ValueError("ORT returned an unexpected tensor count")
    outputs = dict(zip(computed, values))
    return [
        passthrough[name].copy() if name in passthrough else outputs[name]
        for name in names
    ]
