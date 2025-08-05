import unittest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../arceus')))
from core import init

class TestCore(unittest.TestCase):
    def test_init_host(self):
        """Test the init function in host mode."""
        result = init(mode='host')
        self.assertIsNotNone(result)

    def test_init_join(self):
        """Test the init function in join mode."""
        result = init(mode='join')
        self.assertIsNotNone(result)

if __name__ == '__main__':
    unittest.main()