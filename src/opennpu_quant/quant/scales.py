import numpy as np

MIN_EXPONENT = -126
MAX_EXPONENT = 127
MSE_BLOCK_ELEMENTS = 1 << 20


def normal_scale(value):
    """Reject unrepresentable/subnormal scales before converting to FP32."""
    value = np.asarray(value, np.float64)
    info = np.finfo(np.float32)
    if (
        not np.isfinite(value).all()
        or np.any(value < info.tiny)
        or np.any(value > info.max)
    ):
        raise ValueError("Scale must be a positive FP32 normal value")
    result = value.astype(np.float32)
    if not np.isfinite(result).all() or np.any(result < info.tiny):
        raise ValueError("Scale cannot be represented as a FP32 normal value")
    return result


def _power(exponent):
    exponent = np.asarray(exponent)
    if np.any(exponent < MIN_EXPONENT) or np.any(exponent > MAX_EXPONENT):
        raise ValueError("PoT exponent outside FP32 normal range [-126, 127]")
    return np.asarray(
        np.ldexp(np.ones(exponent.shape, np.float64), exponent.astype(np.int32)),
        np.float32,
    )


def pot_scale(value, policy):
    normal_scale(value)
    logarithm = np.log2(np.asarray(value, np.float64))
    if policy == "pot_nearest":
        exponent = np.rint(logarithm)
    elif policy == "pot_ceil":
        exponent = np.ceil(logarithm)
    else:
        raise ValueError("Expected pot_nearest or pot_ceil")
    return _power(exponent)


def pot_candidates(value):
    """Distinct representable floor/ceil candidates for one scalar base scale."""
    normal_scale(value)
    logarithm = float(np.log2(np.float64(value)))
    exponents = sorted({int(np.floor(logarithm)), int(np.ceil(logarithm))})
    result = [float(_power(e)) for e in exponents if MIN_EXPONENT <= e <= MAX_EXPONENT]
    if not result:
        raise ValueError("No FP32 normal PoT candidate")
    return result


def is_pot(value):
    value = np.asarray(value)
    mantissa, _ = np.frexp(value)
    return bool(
        np.isfinite(value).all()
        and np.all(value >= np.finfo(np.float32).tiny)
        and np.all(value <= np.finfo(np.float32).max)
        and np.all(mantissa == 0.5)
    )


def exponents(value):
    return (np.frexp(np.asarray(value))[1] - 1).tolist() if is_pot(value) else None


def activation_parameters(limits, base_scale, base_zero, symmetric, policy):
    low, high = float(limits["lowest"]), float(limits["highest"])
    if not np.isfinite([low, high]).all() or low > high:
        raise ValueError("Invalid activation range")
    scale = base_scale if policy == "float" else pot_scale(base_scale, policy)
    zero = base_zero
    if policy != "float" and not symmetric:
        # Round and clip in FP64 BEFORE narrowing; an INT8 cast alone wraps.
        zero = np.asarray(
            np.clip(np.rint(-128.0 - min(low, 0.0) / float(scale)), -128, 127), np.int8
        )
    qmin = -127 if symmetric else -128
    row = dict(
        policy=policy,
        base_scale=float(base_scale),
        selected_scale=float(scale),
        final_scale=float(scale),
        exponent=exponents(scale),
        zero_point=int(zero),
        selected_range=[low, high],
        range_with_zero=[min(low, 0.0), max(high, 0.0)],
        calibration_integer_range=[qmin, 127],
        grid_range=[
            (-128 - int(zero)) * float(scale),
            (127 - int(zero)) * float(scale),
        ],
    )
    return scale, zero, row


def candidate_mse(weight, scale, encode):
    """Score the actual stored INT8 and FP32 DQ values using bounded temporaries."""
    total, count = np.float64(0), 0
    iterator = np.nditer(
        weight,
        flags=["external_loop", "buffered"],
        op_flags=["readonly"],
        order="C",
        buffersize=MSE_BLOCK_ELEMENTS,
    )
    for block in iterator:
        codes = encode(block, np.asarray(scale, np.float32))
        restored = codes.astype(np.float32) * np.float32(scale)
        difference = block.astype(np.float64) - restored.astype(np.float64)
        total += np.sum(difference * difference, dtype=np.float64)
        count += block.size
    if not count or not np.isfinite(total):
        raise ValueError("Empty or nonfinite weight MSE")
    return float(total / count)


def choose_weight_scale(weight, base_scale, policy, required_scale, encode):
    candidates = (
        pot_candidates(base_scale)
        if policy == "pot_mse"
        else [float(pot_scale(base_scale, policy))]
    )
    scores, cache = [], {}
    effective = {}
    for initial in candidates:
        final = float(pot_scale(max(initial, float(required_scale)), "pot_ceil"))
        if policy == "pot_mse" and final not in cache:
            cache[final] = candidate_mse(weight, final, encode)
        if final in effective:
            effective[final]["initial_candidates"].append(initial)
            continue
        row = dict(
            selected_scale=initial,
            final_scale=final,
            mse=cache.get(final),
            bias_adjusted=final > initial,
            initial_candidates=[initial],
        )
        effective[final] = row
        scores.append(row)
    winner = min(
        scores,
        key=lambda row: (
            row["mse"] if row["mse"] is not None else 0.0,
            row["final_scale"],
            row["selected_scale"],
        ),
    )
    return np.asarray(winner["final_scale"], np.float32), dict(
        base_scale=float(base_scale),
        bias_required_scale=float(required_scale),
        selected_scale=winner["selected_scale"],
        final_scale=winner["final_scale"],
        exponent=exponents(winner["final_scale"]),
        bias_adjusted=winner["bias_adjusted"],
        candidates=scores,
    )
