import os
import types
import time
import builtins
import pytest

import arceus.utils as utils


class DummyObj:
    def __init__(self):
        self.moves = []

    def to(self, device):
        self.moves.append(str(device))
        return self


def test_detect_device_cpu(monkeypatch):
    import torch

    monkeypatch.setattr(torch, 'cuda', types.SimpleNamespace(is_available=lambda: False), raising=False)
    # ensure mps path also false
    monkeypatch.setattr(torch, 'backends', types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False)), raising=False)

    device, info = utils.detect_device()
    assert str(device) == 'cpu'
    assert 'CPU' in info


def test_get_device_backend(monkeypatch):
    import torch
    monkeypatch.setattr(torch, 'cuda', types.SimpleNamespace(is_available=lambda: True), raising=False)
    assert utils.get_device_backend() == 'nccl'
    monkeypatch.setattr(torch, 'cuda', types.SimpleNamespace(is_available=lambda: False), raising=False)
    assert utils.get_device_backend() == 'gloo'


def test_move_to_device_success():
    obj = DummyObj()
    dev = types.SimpleNamespace(type='cpu')
    out = utils.move_to_device(obj, dev)
    assert out is obj
    assert obj.moves[-1].endswith("cpu")


def test_move_to_device_fallback_cpu(monkeypatch, capsys):
    class BadObj:
        def __init__(self):
            self.moves = []
        def to(self, device):
            if str(device) != 'cpu':
                raise RuntimeError('boom')
            self.moves.append('cpu')
            return self
    obj = BadObj()
    dev = 'cuda:0'
    out = utils.move_to_device(obj, dev)
    captured = capsys.readouterr().out
    assert 'Warning' in captured
    assert out is obj
    assert obj.moves == ['cpu']


def test_print_device_info(capsys):
    utils.print_device_info('cpu', 'CPU (4 cores)', rank=0)
    out = capsys.readouterr().out
    assert 'Using device: CPU' in out
    assert 'Rank 0' in out


def test_banner(capsys):
    utils.banner('hello world')
    out = capsys.readouterr().out
    assert 'hello world' in out


def test_wait_for_sessions_immediate():
    class Beacon:
        def get_active_sessions(self):
            return {'ABCD': ('1.2.3.4', 1234)}
    sessions = utils.wait_for_sessions(Beacon(), timeout=0.1)
    assert sessions


def test_wait_for_sessions_timeout():
    class Beacon:
        def __init__(self):
            self.calls = 0
        def get_active_sessions(self):
            self.calls += 1
            return {}
    start = time.time()
    sessions = utils.wait_for_sessions(Beacon(), timeout=0.2)
    assert sessions == {}
    assert time.time() - start >= 0.2


def test_pick_session(monkeypatch, capsys):
    sessions = {
        'S1': ('10.0.0.1', 1),
        'S2': ('10.0.0.2', 2),
    }
    # Choose 2nd option; simulate bad then good input
    inputs = iter(['x', '2'])
    monkeypatch.setattr(builtins, 'input', lambda _: next(inputs))
    picked = utils.pick_session(sessions)
    assert picked in sessions


@pytest.mark.parametrize('system', ['Linux', 'Darwin'])
def test_setup_macos_gloo_env_safe(monkeypatch, system):
    # Ensure function returns None for non-Darwin and sets env for Darwin
    monkeypatch.setenv('GLOO_SOCKET_IFNAME', '', raising=False)
    monkeypatch.setenv('GLOO_SOCKET_IFADDR', '', raising=False)

    import platform
    monkeypatch.setattr(platform, 'system', lambda: system)

    # Stub subprocess and socket usage inside selection path
    import subprocess
    class Result:
        def __init__(self, out):
            self.stdout = out
    def fake_run(cmd, capture_output=False, text=False, check=False):
        assert cmd[:2] == ['ipconfig', 'getifaddr']
        return Result('192.168.1.10')
    monkeypatch.setattr(subprocess, 'run', fake_run)

    if system == 'Darwin':
        ip = utils.setup_macos_gloo_env()
        assert os.getenv('GLOO_SOCKET_IFNAME')
        assert os.getenv('GLOO_SOCKET_IFADDR')
        assert ip
    else:
        assert utils.setup_macos_gloo_env() is None


def test_validate_gloo_setup_skipped(monkeypatch):
    # avoid initializing torch.distributed by mocking dist
    import torch
    class FakeDist:
        def init_process_group(self, *a, **k):
            raise RuntimeError('skip init in tests')
    monkeypatch.setattr(torch, 'distributed', FakeDist(), raising=False)
    ok = utils.validate_gloo_setup()
    assert ok is False