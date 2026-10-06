from types import SimpleNamespace
import unittest
from unittest.mock import Mock
import numpy as np
from opennpu_quant.ort.fetch import run_tensors


class FetchTests(unittest.TestCase):
    def test_computed_outputs_cannot_be_overridden_by_extra_feeds(self):
        session = Mock()
        session.get_inputs.return_value = [SimpleNamespace(name="x")]
        session.run.side_effect = lambda names, feed: [feed["x"] * 2 for _ in names]
        x = np.arange(12, dtype=np.float32).reshape(3, 4).T
        y, a, b = run_tensors(session, ["y", "x", "x"], dict(x=x, y=x * 999))
        self.assertEqual(["y"], session.run.call_args.args[0])
        self.assertEqual(["x"], list(session.run.call_args.args[1]))
        np.testing.assert_array_equal(x * 2, y)
        np.testing.assert_array_equal(x, a)
        self.assertFalse(np.shares_memory(x, a))
        self.assertFalse(np.shares_memory(a, b))
        a.fill(-1)
        self.assertEqual(11, x.max())
        self.assertEqual(11, b.max())

    def test_passthrough_only_never_runs_backend(self):
        session = Mock()
        session.get_inputs.return_value = [SimpleNamespace(name="x")]
        x = np.array([1, 2], dtype=np.float32)
        np.testing.assert_array_equal(x, run_tensors(session, ["x"], dict(x=x))[0])
        session.run.assert_not_called()

    def test_missing_input_and_backend_error_are_not_hidden(self):
        session = Mock()
        session.get_inputs.return_value = [SimpleNamespace(name="x")]
        with self.assertRaises(KeyError):
            run_tensors(session, ["x"], {})
        session.run.side_effect = RuntimeError("backend failure")
        with self.assertRaisesRegex(RuntimeError, "backend failure"):
            run_tensors(session, ["y"], dict(x=np.ones(1)))

    def test_cpu_graph_with_passthrough_and_computed_output(self):
        from onnx import helper as h
        from opennpu_quant.ort.session import create_session
        from opennpu_quant.ort.config import OrtConfig

        m = h.make_model(
            h.make_graph(
                [h.make_node("Relu", ["x"], ["y"])],
                "fetch",
                [h.make_tensor_value_info("x", 1, [2])],
                [h.make_tensor_value_info(n, 1, [2]) for n in ("x", "y")],
            ),
            opset_imports=[h.make_opsetid("", 17)],
            ir_version=10,
        )
        session = create_session(m, OrtConfig.cpu())
        x = np.array([-1, 2], np.float32)
        a, b = run_tensors(session, ["x", "y"], dict(x=x))
        np.testing.assert_array_equal(a, x)
        np.testing.assert_array_equal(b, [0, 2])

    def test_calibration_and_adaround_use_the_shared_helper(self):
        from tests.library_fixtures import toy_model, feeds
        from opennpu_quant import calibrate, CalibrationConfig, OrtConfig
        from opennpu_quant.quant import calibration, adaround
        from unittest.mock import patch

        with patch.object(calibration, "run_tensors", wraps=run_tensors) as fetched:
            calibrate(
                toy_model(), feeds, config=CalibrationConfig(), ort=OrtConfig.cpu()
            )
            self.assertEqual(8, fetched.call_count)
        session = Mock()
        session.get_inputs.return_value = [SimpleNamespace(name="x")]
        with patch("opennpu_quant.ort.fetch.run_tensors", wraps=run_tensors) as fetched:
            adaround._run(session, ["x"], dict(x=np.ones(1)))
            fetched.assert_called_once()


if __name__ == "__main__":
    unittest.main()
