import unittest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../arceus')))
from distributed import TrainingHost

class TestDistributed(unittest.TestCase):
    def test_training_host_initialization(self):
        """Test that TrainingHost initializes properly."""
        host = TrainingHost()
        self.assertIsNotNone(host)

    def test_training_host_accept_connection(self):
        """Test that TrainingHost can accept connections."""
        host = TrainingHost()
        result = host.accept_connection()
        self.assertTrue(result)

if __name__ == '__main__':
    unittest.main()