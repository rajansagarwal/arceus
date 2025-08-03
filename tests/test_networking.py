import os
import sys
import unittest
from unittest import mock

# Ensure repository root on sys.path when running directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))


class TestNetworking(unittest.TestCase):
    def setUp(self):
        global net
        from arceus import networking as _net
        net = _net

    def test_get_broadcast_ip(self):
        with mock.patch("arceus.networking.get_local_ip", return_value="192.168.1.42"):
            bcast = net.get_broadcast_ip()
            self.assertIn(bcast, ("192.168.1.255", "192.168.1.255"))

    def test_find_free_port(self):
        port = net.find_free_port()
        self.assertIsInstance(port, int)
        self.assertTrue(1024 <= port <= 65535)


if __name__ == "__main__":
    unittest.main()
