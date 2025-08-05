import unittest
from unittest.mock import patch, MagicMock
from arceus.distributed import TrainingHost

class TestDistributed(unittest.TestCase):
    
    @patch('arceus.distributed.get_local_ip', return_value='127.0.0.1')
    @patch('arceus.distributed.find_free_port', return_value=12345)
    @patch('socket.socket')
    def test_training_host_start(self, mock_socket, mock_find_free_port, mock_get_local_ip):
        """Test that TrainingHost can start a training session and accept connections."""
        mock_sock_instance = MagicMock()
        mock_socket.return_value = mock_sock_instance
        
        host = TrainingHost('session123', 29500)
        world = host.start_training()
        
        self.assertIn(('127.0.0.1', 29500), [address for _, address in world])
        mock_sock_instance.bind.assert_called_with(('', 12345))
        mock_sock_instance.listen.assert_called_once()

if __name__ == '__main__':
    unittest.main()