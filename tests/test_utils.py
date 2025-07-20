import unittest
from unittest.mock import patch, MagicMock, call
import os
import sys
import subprocess
import ipaddress
import socket

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from arceus.utils import (
    _pick_macos_iface, setup_macos_gloo_env, validate_gloo_setup,
    init_pytorch_distributed
)


class TestMacOSNetworkInterface(unittest.TestCase):
    """Test macOS network interface selection"""
    
    @patch('platform.system')
    def test_pick_macos_iface_non_darwin(self, mock_platform):
        """Test interface selection on non-macOS systems"""
        mock_platform.return_value = 'Linux'
        
        result = _pick_macos_iface()
        
        self.assertEqual(result, ("en0", "127.0.0.1"))
    
    @patch('platform.system')
    @patch('subprocess.check_output')
    @patch('socket.socket')
    def test_pick_macos_iface_success(self, mock_socket, mock_subprocess, mock_platform):
        """Test successful interface selection on macOS"""
        mock_platform.return_value = 'Darwin'
        
        ifconfig_output = """
en0: flags=8863<UP,BROADCAST,SMART,RUNNING,SIMPLEX,MULTICAST> mtu 1500
	inet 192.168.1.100 netmask 0xffffff00 broadcast 192.168.1.255
en1: flags=8822<BROADCAST,SMART,SIMPLEX,MULTICAST> mtu 1500
	inet 10.0.0.100 netmask 0xffffff00 broadcast 10.0.0.255
"""
        mock_subprocess.return_value = ifconfig_output.encode()
        
        mock_sock = MagicMock()
        mock_socket.return_value = mock_sock
        
        result = _pick_macos_iface()
        
        self.assertEqual(result, ("en0", "192.168.1.100"))
        mock_sock.bind.assert_called_with(("192.168.1.100", 0))
        mock_sock.close.assert_called_once()
    
    @patch('platform.system')
    @patch('subprocess.check_output')
    @patch('socket.socket')
    def test_pick_macos_iface_skip_cgn(self, mock_socket, mock_subprocess, mock_platform):
        """Test skipping carrier-grade NAT addresses"""
        mock_platform.return_value = 'Darwin'
        
        ifconfig_output = """
en0: flags=8863<UP,BROADCAST,SMART,RUNNING,SIMPLEX,MULTICAST> mtu 1500
	inet 100.64.1.100 netmask 0xffffff00 broadcast 100.64.1.255
en1: flags=8863<UP,BROADCAST,SMART,RUNNING,SIMPLEX,MULTICAST> mtu 1500
	inet 192.168.1.100 netmask 0xffffff00 broadcast 192.168.1.255
"""
        mock_subprocess.return_value = ifconfig_output.encode()
        
        mock_sock = MagicMock()
        mock_socket.return_value = mock_sock
        
        result = _pick_macos_iface()
        
        self.assertEqual(result, ("en1", "192.168.1.100"))
    
    @patch('platform.system')
    @patch('subprocess.check_output')
    @patch('subprocess.run')
    def test_pick_macos_iface_fallback(self, mock_run, mock_subprocess, mock_platform):
        """Test fallback to en0 when interface detection fails"""
        mock_platform.return_value = 'Darwin'
        mock_subprocess.side_effect = Exception("ifconfig failed")
        
        mock_result = MagicMock()
        mock_result.stdout.strip.return_value = "192.168.1.100"
        mock_run.return_value = mock_result
        
        result = _pick_macos_iface()
        
        self.assertEqual(result, ("en0", "192.168.1.100"))
        mock_run.assert_called_with(
            ["ipconfig", "getifaddr", "en0"],
            capture_output=True, text=True, check=True
        )


class TestMacOSGlooSetup(unittest.TestCase):
    """Test macOS Gloo environment setup"""
    
    def setUp(self):
        """Set up test environment"""
        self.original_env = {}
        gloo_vars = [k for k in os.environ.keys() if k.startswith('GLOO_')]
        for var in gloo_vars:
            self.original_env[var] = os.environ.pop(var, None)
    
    def tearDown(self):
        """Restore original environment"""
        gloo_vars = [k for k in os.environ.keys() if k.startswith('GLOO_')]
        for var in gloo_vars:
            os.environ.pop(var, None)
        
        for var, value in self.original_env.items():
            if value is not None:
                os.environ[var] = value
    
    @patch('platform.system')
    def test_setup_macos_gloo_env_non_darwin(self, mock_platform):
        """Test Gloo setup on non-macOS systems"""
        mock_platform.return_value = 'Linux'
        
        result = setup_macos_gloo_env()
        
        self.assertIsNone(result)
    
    @patch('platform.system')
    @patch('arceus.utils._pick_macos_iface')
    def test_setup_macos_gloo_env_success(self, mock_pick_iface, mock_platform):
        """Test successful Gloo environment setup"""
        mock_platform.return_value = 'Darwin'
        mock_pick_iface.return_value = ("en0", "192.168.1.100")
        
        result = setup_macos_gloo_env()
        
        self.assertEqual(result, "192.168.1.100")
        
        self.assertEqual(os.environ["GLOO_SOCKET_IFNAME"], "en0")
        self.assertEqual(os.environ["GLOO_SOCKET_IFADDR"], "192.168.1.100")
        self.assertEqual(os.environ["GLOO_SOCKET_FAMILY"], "AF_INET")
        self.assertEqual(os.environ["GLOO_SOCKET_DISABLE_IPV6"], "1")
        self.assertEqual(os.environ["GLOO_SOCKET_FORCE_IPV4"], "1")
        self.assertEqual(os.environ["GLOO_ALLOW_UNSECURED"], "1")
    
    @patch('platform.system')
    @patch('os.getenv')
    @patch('subprocess.run')
    def test_setup_macos_gloo_env_user_interface(self, mock_run, mock_getenv, mock_platform):
        """Test Gloo setup with user-specified interface"""
        mock_platform.return_value = 'Darwin'
        mock_getenv.return_value = "en1"  # User specified interface
        
        mock_result = MagicMock()
        mock_result.stdout.strip.return_value = "10.0.0.100"
        mock_run.return_value = mock_result
        
        result = setup_macos_gloo_env()
        
        self.assertEqual(result, "10.0.0.100")
        self.assertEqual(os.environ["GLOO_SOCKET_IFNAME"], "en1")
        self.assertEqual(os.environ["GLOO_SOCKET_IFADDR"], "10.0.0.100")
    
    @patch('platform.system')
    @patch('arceus.utils._pick_macos_iface')
    def test_setup_macos_gloo_env_fallback(self, mock_pick_iface, mock_platform):
        """Test Gloo setup fallback when interface detection fails"""
        mock_platform.return_value = 'Darwin'
        mock_pick_iface.return_value = (None, None)  # Interface detection failed
        
        with patch('builtins.print') as mock_print:
            result = setup_macos_gloo_env()
            
            warning_printed = any("Warning: Could not determine valid interface/IP" in str(call) for call in mock_print.call_args_list)
            success_printed = any("macOS Gloo pinned to en0 (127.0.0.1)" in str(call) for call in mock_print.call_args_list)
            self.assertTrue(warning_printed or success_printed)
        
        self.assertEqual(result, "127.0.0.1")
        self.assertEqual(os.environ["GLOO_SOCKET_IFNAME"], "en0")
        self.assertEqual(os.environ["GLOO_SOCKET_IFADDR"], "127.0.0.1")


class TestIPv6Handling(unittest.TestCase):
    """Test IPv6 interference handling"""
    
    @patch('torch.distributed.init_process_group')
    @patch('torch.distributed.destroy_process_group')
    def test_validate_gloo_setup_success(self, mock_destroy, mock_init):
        """Test successful Gloo validation"""
        mock_init.return_value = None  # Successful initialization
        
        with patch('builtins.print') as mock_print:
            result = validate_gloo_setup()
            
            mock_print.assert_called_with("✓ Gloo single-rank validation passed")
        
        self.assertTrue(result)
        mock_init.assert_called_once()
        mock_destroy.assert_called_once()
    
    @patch('torch.distributed.init_process_group')
    def test_validate_gloo_setup_failure(self, mock_init):
        """Test Gloo validation failure"""
        mock_init.side_effect = RuntimeError("IPv6 error")
        
        with patch('builtins.print') as mock_print:
            result = validate_gloo_setup()
            
            mock_print.assert_called_with("✗ Gloo validation failed: IPv6 error")
        
        self.assertFalse(result)
    
    @patch('torch.distributed.init_process_group')
    @patch('arceus.utils.setup_macos_gloo_env')
    @patch('platform.system')
    def test_init_pytorch_distributed_ipv6_retry(self, mock_platform, mock_setup, mock_init):
        """Test IPv6 error handling with retry logic"""
        mock_platform.return_value = 'Darwin'
        mock_setup.return_value = "192.168.1.100"
        
        mock_init.side_effect = [
            RuntimeError("fe80:: link-local address error"),
            None  # Success on retry
        ]
        
        world = [("host-uuid", ("192.168.1.100", 29500))]
        rank = 0
        
        with patch('builtins.print') as mock_print:
            init_pytorch_distributed(world, rank)
            
            print_calls = [call[0][0] for call in mock_print.call_args_list]
            ipv6_error_detected = any("IPv6 link-local address detected" in call for call in print_calls)
            retry_attempted = any("Retrying with stronger IPv6 disabling" in call for call in print_calls)
            
            self.assertTrue(ipv6_error_detected)
            self.assertTrue(retry_attempted)
        
        self.assertEqual(os.environ["GLOO_SOCKET_FORCE_IPV4"], "1")
        self.assertEqual(os.environ["GLOO_SOCKET_PREFER_IPV4"], "1")
    
    @patch('torch.distributed.init_process_group')
    @patch('arceus.utils.setup_macos_gloo_env')
    @patch('platform.system')
    def test_init_pytorch_distributed_ipv6_failure(self, mock_platform, mock_setup, mock_init):
        """Test IPv6 error handling when retries fail"""
        mock_platform.return_value = 'Darwin'
        mock_setup.return_value = "192.168.1.100"
        
        mock_init.side_effect = RuntimeError("fe80:: persistent IPv6 error")
        
        world = [("host-uuid", ("192.168.1.100", 29500))]
        rank = 0
        
        with self.assertRaises(RuntimeError):
            with patch('builtins.print'):
                init_pytorch_distributed(world, rank)
    
    @patch('torch.distributed.init_process_group')
    @patch('arceus.utils.setup_macos_gloo_env')
    @patch('platform.system')
    def test_init_pytorch_distributed_timeout_guidance(self, mock_platform, mock_setup, mock_init):
        """Test timeout error with cross-device troubleshooting guidance"""
        mock_platform.return_value = 'Darwin'
        mock_setup.return_value = "192.168.1.100"
        
        mock_init.side_effect = RuntimeError("timeout after 30 seconds")
        
        world = [("host-uuid", ("192.168.1.100", 29500))]
        rank = 0
        
        with self.assertRaises(RuntimeError) as context:
            with patch('builtins.print') as mock_print:
                init_pytorch_distributed(world, rank)
                
                print_calls = [call[0][0] for call in mock_print.call_args_list]
                guidance_provided = any("Common fixes for cross-device communication" in call for call in print_calls)
                self.assertTrue(guidance_provided)
        
        self.assertIn("distributed training initialization timed out", str(context.exception))


if __name__ == '__main__':
    unittest.main()
