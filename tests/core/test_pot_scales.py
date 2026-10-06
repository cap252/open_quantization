from dataclasses import replace
import json
import unittest
import numpy as np
from opennpu_quant.quant.config import QuantizationConfig
from tests.library_fixtures import quantization_matrix
from opennpu_quant.quant.scales import (
    pot_scale,
    pot_candidates,
    is_pot,
    activation_parameters,
    candidate_mse,
    choose_weight_scale,
)


class ScaleTests(unittest.TestCase):
    def test_configuration_roundtrip_defaults_and_distinct_names(self):
        for old in quantization_matrix():
            minimal = old.to_dict()
            minimal.pop("activation_scale_policy")
            minimal.pop("weight_scale_policy")
            self.assertEqual(old, QuantizationConfig.from_dict(minimal))
            self.assertNotIn("__as_", old.name)
            configured = replace(
                old,
                activation_scale_policy="pot_nearest",
                weight_scale_policy="pot_mse",
            )
            self.assertEqual(
                configured,
                QuantizationConfig.from_dict(
                    json.loads(json.dumps(configured.to_dict()))
                ),
            )
            self.assertNotEqual(old.name, configured.name)
            self.assertEqual("pot_mse", configured.scheme()["weight_scale_policy"])
        for kwargs in (
            {"activation_scale_policy": "pot_mse"},
            {"weight_scale_policy": "unknown"},
        ):
            with self.assertRaises(ValueError):
                QuantizationConfig(**kwargs)

    def test_exponents_power_boundaries_and_logarithmic_ties(self):
        for exponent in (-126, -7, 0, 7, 127):
            value = np.float32(2.0**exponent)
            self.assertEqual(value, pot_scale(value, "pot_nearest"))
            self.assertEqual(value, pot_scale(value, "pot_ceil"))
            self.assertTrue(is_pot(value))
        for exponent in (-7, 0, 7):
            value = np.float32(2.0**exponent)
            below = np.nextafter(value, np.float32(0))
            above = np.nextafter(value, np.float32(np.inf))
            self.assertEqual(value, pot_scale(below, "pot_ceil"))
            self.assertEqual(value * 2, pot_scale(above, "pot_ceil"))
            self.assertEqual(value, pot_scale(above, "pot_nearest"))
        for exponent in (-4, -3, 2, 3):
            self.assertEqual(
                2.0 ** round(exponent + 0.5),
                pot_scale(2.0 ** (exponent + 0.5), "pot_nearest"),
            )
        self.assertFalse(is_pot(np.nextafter(np.float32(1), np.float32(2))))
        self.assertEqual([0.5, 1.0], pot_candidates(0.7))

    def test_nonfinite_zero_and_unrepresentable_values_fail(self):
        for value in (
            0.0,
            -1.0,
            np.nan,
            np.inf,
            np.nextafter(np.float32(0), np.float32(1)),
        ):
            with self.assertRaises(ValueError):
                pot_scale(value, "pot_nearest")
        with self.assertRaises(ValueError):
            pot_scale(np.finfo(np.float32).max, "pot_ceil")
        self.assertEqual([2.0**127], pot_candidates(np.finfo(np.float32).max))

    def test_asymmetric_zero_clipping_before_int8_conversion(self):
        for low, high, expected in [(0, 1, -128), (-1, 0, 0), (-100.25, 154.75, -28)]:
            base = np.float32((max(high, 0) - min(low, 0)) / 255)
            scale, zero, row = activation_parameters(
                {"lowest": low, "highest": high}, base, np.int8(0), False, "pot_ceil"
            )
            self.assertEqual(expected, int(zero))
            self.assertTrue(is_pot(scale))
            self.assertEqual(int(zero), row["zero_point"])
        # Nearest can narrow the domain enough that a raw zero point would wrap.
        _, zero, _ = activation_parameters(
            {"lowest": -1.0, "highest": -0.1},
            np.float32(1 / 255),
            np.int8(127),
            False,
            "pot_nearest",
        )
        self.assertEqual(127, int(zero))
        for low, high in ((np.nan, 1), (2, 1)):
            with self.assertRaises(ValueError):
                activation_parameters(
                    {"lowest": low, "highest": high},
                    np.float32(1),
                    np.int8(0),
                    False,
                    "pot_ceil",
                )

    def test_mse_uses_bounded_actual_code_blocks(self):
        weight = np.linspace(-3, 3, (1 << 20) + 17, dtype=np.float32)
        lengths = []

        def encode(values, scale):
            lengths.append(values.size)
            return np.clip(np.rint(values / scale), -128, 127).astype(np.int8)

        actual = candidate_mse(weight, np.float32(0.03125), encode)
        expected = np.mean(
            (
                weight.astype(np.float64)
                - np.clip(np.rint(weight / 0.03125), -128, 127).astype(np.float32)
                * np.float32(0.03125)
            )
            ** 2
        )
        self.assertAlmostEqual(expected, actual, places=14)
        self.assertLessEqual(max(lengths), 1 << 20)

    def test_mse_tie_and_bias_candidate_deduplication(self):
        def encode(values, scale):
            return np.clip(np.rint(values / scale), -128, 127).astype(np.int8)

        weight = np.zeros(16, np.float32)
        scale, row = choose_weight_scale(weight, 0.7, "pot_mse", 0.5, encode)
        self.assertEqual(0.5, float(scale))
        self.assertEqual(2, len(row["candidates"]))
        scale, row = choose_weight_scale(weight, 0.7, "pot_mse", 1.1, encode)
        self.assertEqual(2.0, float(scale))
        self.assertEqual(1, len(row["candidates"]))
        self.assertEqual([0.5, 1.0], row["candidates"][0]["initial_candidates"])
        self.assertTrue(row["bias_adjusted"])
