import unittest
from unittest.mock import patch, MagicMock, call
import json
import socket
import time
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from arceus.networking import (
    get_local_ip, get_broadcast_ip, find_free_port,
    DISCOVERY_PORT, MAGIC_HEADER, BROADCAST_INTERVAL
)


class TestNetworkingFunctions(unittest.TestCase):
    """Test basic networking utility functions"""
    
    @patch('socket.socket')
    def test_get_local_ip(self, mock_socket):
        """Test getting local IP address via Google DNS"""
        mock_sock = MagicMock()
        mock_socket.return_value = mock_sock
        mock_sock.getsockname.return_value = ('192.168.1.100', 12345)
        
        result = get_local_ip()
        
        self.assertEqual(result, '192.168.1.100')
        mock_sock.connect.assert_called_once_with(("8.8.8.8", 80))
        mock_sock.close.assert_called_once()
    
    
    def test_get_broadcast_ip_with_netifaces(self):
        """Test broadcast IP detection with netifaces available"""
        mock_netifaces = MagicMock()
        mock_netifaces.interfaces.return_value = ['en0', 'lo0']
        mock_netifaces.AF_INET = 2
        mock_netifaces.ifaddresses.return_value = {
            2: [{'addr': '192.168.1.100', 'broadcast': '192.168.1.255'}]
        }
        
        mock_ipaddress = MagicMock()
        mock_ip = MagicMock()
        mock_ip.is_loopback = False
        mock_ip.is_link_local = False
        mock_ipaddress.IPv4Address.return_value = mock_ip
        
        def mock_import(name, *args, **kwargs):
            if name == 'netifaces':
                return mock_netifaces
            elif name == 'ipaddress':
                return mock_ipaddress
            return __import__(name, *args, **kwargs)
        
        with patch('builtins.__import__', side_effect=mock_import):
            result = get_broadcast_ip()
            
        self.assertEqual(result, '192.168.1.255')
    
    @patch('socket.socket')
    def test_find_free_port(self, mock_socket):
        """Test finding a free port"""
        mock_sock = MagicMock()
        mock_socket.return_value = mock_sock
        mock_sock.getsockname.return_value = ('0.0.0.0', 12345)
        
        result = find_free_port()
        
        self.assertEqual(result, 12345)
        mock_sock.bind.assert_called_once_with(("", 0))


class TestBroadcastIPDiscovery(unittest.TestCase):
    """Test broadcast IP discovery for cross-device networking"""
    
    
    def test_get_all_broadcast_ips_with_netifaces(self):
        """Test broadcast IP discovery with multiple network interfaces"""
        from arceus.networking import UDPBeacon
        
        beacon = object.__new__(UDPBeacon)
        beacon.session_id = "test-session"
        beacon.tcp_port = 8080
        
        mock_netifaces = MagicMock()
        mock_netifaces.interfaces.return_value = ['en0', 'en1', 'bridge0']
        mock_netifaces.AF_INET = 2
        
        def mock_ifaddresses(interface):
            if interface == 'en0':
                return {2: [{'addr': '192.168.1.100', 'broadcast': '192.168.1.255'}]}
            elif interface == 'en1':
                return {2: [{'addr': '10.0.0.100', 'broadcast': '10.0.0.255'}]}
            elif interface == 'bridge0':
                return {2: [{'addr': '172.16.0.100', 'broadcast': '172.16.0.255'}]}
            return {}
        
        mock_netifaces.ifaddresses.side_effect = mock_ifaddresses
        
        mock_ipaddress = MagicMock()
        mock_ip = MagicMock()
        mock_ip.is_loopback = False
        mock_ip.is_link_local = False
        mock_ipaddress.IPv4Address.return_value = mock_ip
        
        def mock_import(name, *args, **kwargs):
            if name == 'netifaces':
                return mock_netifaces
            elif name == 'ipaddress':
                return mock_ipaddress
            return __import__(name, *args, **kwargs)
        
        with patch('builtins.__import__', side_effect=mock_import):
            result = beacon._get_all_broadcast_ips()
            
        expected = ['192.168.1.255', '10.0.0.255', '172.16.0.255']
        self.assertEqual(result, expected)


class TestCrossDeviceNetworking(unittest.TestCase):
    """Test cross-device networking improvements"""
    
    def test_multiple_network_interfaces_get_broadcast_ip(self):
        """Test handling multiple network interfaces for cross-device discovery"""
        mock_netifaces = MagicMock()
        mock_netifaces.interfaces.return_value = ['en0', 'en1', 'bridge0']
        mock_netifaces.AF_INET = 2
        
        def mock_ifaddresses_func(interface):
            if interface == 'en0':
                return {2: [{'addr': '192.168.1.100', 'broadcast': '192.168.1.255'}]}
            elif interface == 'en1':
                return {2: [{'addr': '10.0.0.100', 'broadcast': '10.0.0.255'}]}
            elif interface == 'bridge0':
                return {2: [{'addr': '172.16.0.100', 'broadcast': '172.16.0.255'}]}
            return {}
        
        mock_netifaces.ifaddresses.side_effect = mock_ifaddresses_func
        
        mock_ipaddress = MagicMock()
        mock_ip = MagicMock()
        mock_ip.is_loopback = False
        mock_ip.is_link_local = False
        mock_ipaddress.IPv4Address.return_value = mock_ip
        
        def mock_import(name, *args, **kwargs):
            if name == 'netifaces':
                return mock_netifaces
            elif name == 'ipaddress':
                return mock_ipaddress
            return __import__(name, *args, **kwargs)
        
        with patch('builtins.__import__', side_effect=mock_import):
            result = get_broadcast_ip()
            
        self.assertEqual(result, '192.168.1.255')
    
    def test_debug_logging_functionality(self):
        """Test debug logging for failed broadcasts"""
        import os
        from arceus.networking import UDPBeacon
        
        beacon = object.__new__(UDPBeacon)
        beacon.session_id = "test-session"
        beacon.tcp_port = 8080
        beacon.running = True  # Allow one iteration
        
        mock_sock = MagicMock()
        mock_sock.sendto.side_effect = OSError("Network unreachable")
        beacon.sock = mock_sock
        
        with patch('os.getenv', return_value="1"):  # ARCEUS_DEBUG=1
            with patch('builtins.print') as mock_print:
                with patch.object(beacon, '_get_all_broadcast_ips', return_value=['192.168.1.255']):
                    with patch('arceus.networking.get_local_ip', return_value='192.168.1.100'):
                        with patch('time.sleep') as mock_sleep:
                            def stop_after_first_call(*args):
                                beacon.running = False
                            mock_sleep.side_effect = stop_after_first_call
                            
                            beacon._broadcast_loop()
                            
                            mock_print.assert_called_with("Failed to broadcast to 192.168.1.255: Network unreachable")


if __name__ == '__main__':
    unittest.main()
