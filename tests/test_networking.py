import types
import socket
import arceus.networking as net


def test_get_broadcast_ip(monkeypatch):
    monkeypatch = monkeypatch  # for pytest

    monkeypatch.setattr(net, 'get_local_ip', lambda: '10.0.1.23')
    assert net.get_broadcast_ip() == '10.0.1.255'


def test_find_free_port():
    port = net.find_free_port()
    assert isinstance(port, int)
    assert 0 < port < 65536