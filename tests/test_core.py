import os
import sys
import io
import unittest
from unittest import mock

# Ensure repository root on sys.path when running directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))


class TestCore(unittest.TestCase):
    def setUp(self):
        global core
        from arceus import core as _core
        core = _core

    def test_print_model_summary(self):
        import torch
        import torch.nn as nn
        model = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))
        buf = io.StringIO()
        with mock.patch("sys.stdout", buf):
            core._print_model_summary(model)
        out = buf.getvalue()
        self.assertIn("Parameters", out)

    def test_wrap_single_process_auto_device(self):
        class DummyModel:
            pass
        model = DummyModel()
        with mock.patch.object(core, "_world", [("node", 0)]), \
             mock.patch.object(core, "device", "cpu"), \
             mock.patch("arceus.core.move_to_device", side_effect=lambda m, d: m) as mmove:
            wrapped = core.wrap(model, auto_device=True)
            self.assertIs(wrapped, model)
            mmove.assert_called_once()

    def test_getters_and_to_device(self):
        with mock.patch.object(core, "device", "cpu"):
            self.assertEqual(str(core.get_device()), "cpu")
            info = core.get_device_info()
            self.assertIsInstance(info, str)

        class Dummy:
            def __init__(self):
                self.moved = None
            def to(self, d):
                self.moved = d
                return self
        x = Dummy()
        moved = core.to_device(x)
        self.assertIs(moved, x)
        self.assertEqual(x.moved, core.device)

    def test_get_learning_rate(self):
        class Opt:
            def __init__(self):
                self.param_groups = [{"lr": 0.01}]
        self.assertEqual(core.get_learning_rate(Opt()), 0.01)

    def test_progress_constructs(self):
        with mock.patch("arceus.core.MetricProgressBar") as mpb:
            core.progress(total=10, description="Test")
            mpb.assert_called()


if __name__ == "__main__":
    unittest.main()
