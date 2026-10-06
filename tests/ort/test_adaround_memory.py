import unittest
import numpy as np
import importlib.util

if importlib.util.find_spec("torch"):
    import torch
else:
    torch = None
from opennpu_quant import AdaroundConfig
from opennpu_quant.quant.adaround import _regularizer, _backward_regularizer


@unittest.skipUnless(
    torch is not None, "Optional AdaRound Torch dependency is unavailable"
)
class RegularizerMemoryTests(unittest.TestCase):
    def test_chunked_gradients_match_dense_including_accumulation(self):
        c = AdaroundConfig(steps=10)
        for step in (0, 1, 2, 3, 9):
            for all_inactive in (False, True):
                torch.manual_seed(71)
                a = torch.nn.Parameter(torch.randn(37))
                b = torch.nn.Parameter(a.detach().clone())
                active = (
                    torch.zeros(37, dtype=torch.bool)
                    if all_inactive
                    else torch.rand(37) > 0.4
                )
                # Reconstruction gradient is already accumulated when regularization runs.
                prior = torch.randn(37)
                a.grad = prior.clone()
                b.grad = prior.clone()
                _regularizer(a, active, step, c).backward()
                _backward_regularizer(b, active, step, c, chunk_elements=7, threshold=0)
                torch.testing.assert_close(a.grad, b.grad, atol=1e-8, rtol=2e-7)

    def test_adam_updates_and_final_codes_match(self):
        from opennpu_quant.quant.adaround import rounding_state, rounding_codes

        c = AdaroundConfig(steps=12)
        w = np.random.default_rng(5).normal(size=(7, 9)).astype(np.float32)
        a, fl, sc, active = rounding_state(w, np.float32(0.15), device="cpu")
        b = torch.nn.Parameter(a.detach().clone())
        opts = [torch.optim.Adam([x], lr=0.001) for x in (a, b)]
        for step in range(c.steps):
            for x, opt, chunked in zip((a, b), opts, (False, True)):
                opt.zero_grad(set_to_none=True)
                (
                    rounding_codes(x, fl, active, hard=False) * sc - torch.from_numpy(w)
                ).square().sum().backward()
                if chunked:
                    _backward_regularizer(
                        x, active, step, c, chunk_elements=11, threshold=0
                    )
                else:
                    _regularizer(x, active, step, c).backward()
                opt.step()
            torch.testing.assert_close(a, b, atol=1e-7, rtol=2e-7)
        torch.testing.assert_close(
            rounding_codes(a, fl, active, hard=True),
            rounding_codes(b, fl, active, hard=True),
            atol=0,
            rtol=0,
        )


if __name__ == "__main__":
    unittest.main()
