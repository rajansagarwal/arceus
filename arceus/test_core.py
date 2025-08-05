import unittest
from unittest.mock import patch, MagicMock
from arceus.core import init

class TestCore(unittest.TestCase):
    
    @patch('arceus.core.UDPBeacon')
    @patch('arceus.core.wait_for_sessions')
    @patch('arceus.core.detect_device')
    @patch('arceus.core.print_device_info')
    def test_init_auto_mode_no_sessions(self, mock_print_device_info, mock_detect_device, mock_wait_for_sessions, mock_UDPBeacon):
        """Test init function in auto mode with no available sessions."""
        mock_UDPBeacon.return_value = MagicMock()
        mock_wait_for_sessions.return_value = []  # No sessions available
        mock_detect_device.return_value = ('cpu', 'Generic CPU')
        
        rank, world_size = init(mode='auto')
        
        self.assertEqual(rank, 0)
        self.assertEqual(world_size, 1)
        mock_print_device_info.assert_called_once_with('cpu', 'Generic CPU', 0)

if __name__ == '__main__':
    unittest.main()