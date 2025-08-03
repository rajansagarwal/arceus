import types
import builtins
import pytest
import torch

import arceus.core as core


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(4, 2)
    def forward(self, x):
        return self.lin(x)


def seed_single_process(monkeypatch):
    # Seed internal globals to simulate single-process init
    monkeypatch.setattr(core, '_world', [('uuid', ('localhost', 0))])
    monkeypatch.setattr(core, '_rank', 0)
    monkeypatch.setattr(core, '_device', torch.device('cpu'))
    monkeypatch.setattr(core, '_device_info', 'CPU (test)')


def test_print_model_summary(capsys):
    m = Tiny()
    core._print_model_summary(m)
    out = capsys.readouterr().out
    assert 'model summary' in out.lower()
    assert 'total params' in out


def test_wrap_single_process_moves(monkeypatch):
    seed_single_process(monkeypatch)

    # track move_to_device is called
    called = {'v': 0}
    def fake_move(obj, device):
        called['v'] += 1
        return obj
    monkeypatch.setattr(core, 'move_to_device', fake_move)

    m = Tiny()
    out = core.wrap(m, show_graph=True, auto_device=True)
    assert out is m
    assert called['v'] == 1


def test_progress_constructs(monkeypatch):
    seed_single_process(monkeypatch)

    constructed = {'args': None}
    class FakeMPB:
        def __init__(self, *args):
            constructed['args'] = args
    monkeypatch.setattr(core, 'MetricProgressBar', FakeMPB)

    data = [1, 2, 3]
    core.progress(data)
    args = constructed['args']
    assert args[0] is data
    assert args[1] == 0  # rank


def test_getters_setters(monkeypatch):
    seed_single_process(monkeypatch)
    assert str(core.get_device()) == 'cpu'
    assert 'CPU' in core.get_device_info()

    class Obj:
        def __init__(self):
            self.moved = False
        def to(self, device):
            self.moved = True
            return self
    obj = Obj()
    out = core.to_device(obj)
    assert out is obj and obj.moved


def test_get_learning_rate():
    opt = types.SimpleNamespace(param_groups=[{'lr': 0.005}])
    assert core.get_learning_rate(opt) == 0.005


def test_cli_parsing(monkeypatch):
    # Use auto mode with defaults and ensure it returns tuple
    def fake_parse():
        return 'auto', None, types.SimpleNamespace(timeout=0, port=29500)
    monkeypatch.setattr(core, 'parse_cli_args', fake_parse)

    # Avoid real init; simulate single-process
    def fake_init(mode, session, timeout, port):
        return 0, 1
    monkeypatch.setattr(core, 'init', fake_init)

    rank, world, args = core.cli()
    assert rank == 0 and world == 1