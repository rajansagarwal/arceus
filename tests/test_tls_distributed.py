import os
import unittest
import threading
import time
from unittest.mock import patch, MagicMock

import socket
import arceus
from arceus.distributed import TrainingHost, TrainingJoiner
from arceus.networking import UDPBeacon


class TestTLSDistributed(unittest.TestCase):
    def setUp(self):
        # Save original environment variables
        self.original_tls = os.environ.get("ARCEUS_TLS")
        self.original_verify = os.environ.get("ARCEUS_TLS_VERIFY")
        
        # Set environment variables for testing
        os.environ["ARCEUS_TLS"] = "1"
        os.environ["ARCEUS_TLS_VERIFY"] = "0"
    
    def tearDown(self):
        # Restore original environment variables
        if self.original_tls is None:
            os.environ.pop("ARCEUS_TLS", None)
        else:
            os.environ["ARCEUS_TLS"] = self.original_tls
            
        if self.original_verify is None:
            os.environ.pop("ARCEUS_TLS_VERIFY", None)
        else:
            os.environ["ARCEUS_TLS_VERIFY"] = self.original_verify
    
    def test_beacon_tls_flag(self):
        """Test that the beacon includes TLS flag in broadcast."""
        beacon = UDPBeacon("TEST", 12345, tls_enabled=True)
        try:
            # Give it time to start broadcasting
            time.sleep(0.5)
            
            # Mock the UDP socket to capture broadcasts
            with patch('arceus.networking.socket.socket') as mock_socket:
                # Mock the socket methods
                mock_socket.return_value.sendto.return_value = None
                
                # Create a new beacon to trigger a broadcast
                beacon2 = UDPBeacon("TEST2", 12346, tls_enabled=True)
                
                # Give it time to broadcast
                time.sleep(0.5)
                
                # Check that the TLS flag was included in the broadcast
                calls = mock_socket.return_value.sendto.call_args_list
                found_tls = False
                for call in calls:
                    args, _ = call
                    data = args[0]
                    if b'"tls": true' in data:
                        found_tls = True
                        break
                
                self.assertTrue(found_tls, "TLS flag not found in beacon broadcast")
                
                beacon2.stop()
        finally:
            beacon.stop()
    
    @patch('arceus.distributed.socket.socket')
    def test_host_tls_enabled(self, mock_socket):
        """Test that the host uses TLS when enabled."""
        # Mock socket methods
        mock_server_socket = MagicMock()
        mock_socket.return_value = mock_server_socket
        mock_server_socket.accept.return_value = (MagicMock(), ('127.0.0.1', 12345))
        
        # Mock TLS context
        with patch('arceus.distributed.TLSContext') as mock_tls_context:
            mock_context = MagicMock()
            mock_tls_context.return_value = mock_context
            
            # Create host with TLS enabled
            host = TrainingHost("TEST", 12345, use_tls=True)
            
            # Check that TLS context was created
            mock_tls_context.assert_called_once()
            
            # Clean up
            host.start_training()
    
    @patch('arceus.distributed.socket.socket')
    def test_joiner_tls_enabled(self, mock_socket):
        """Test that the joiner uses TLS when enabled."""
        # Mock socket methods
        mock_client_socket = MagicMock()
        mock_socket.return_value = mock_client_socket
        
        # Mock connection methods
        mock_client_socket.connect.return_value = None
        mock_client_socket.send.return_value = None
        mock_client_socket.recv.return_value = b'{"start": true, "world": [["host", ["127.0.0.1", 12345]]]}'
        
        # Mock TLS context
        with patch('arceus.distributed.TLSContext') as mock_tls_context:
            mock_context = MagicMock()
            mock_tls_context.return_value = mock_context
            
            # Create joiner with TLS enabled
            joiner = TrainingJoiner("127.0.0.1", 12345, use_tls=True)
            
            # Connect to host
            joiner.connect_to_host()
            
            # Check that TLS context was created
            mock_tls_context.assert_called_once()
            
            # Check that socket was wrapped
            mock_context.wrap_socket.assert_called_once()
            
            # Wait for start
            world = joiner.wait_for_start()
            
            # Check world
            self.assertEqual(len(world), 1)
            self.assertEqual(world[0][0], "host")


if __name__ == '__main__':
    unittest.main()