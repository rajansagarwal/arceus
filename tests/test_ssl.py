import os
import socket
import ssl
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import arceus.ssl_utils as ssl_utils
from arceus.ssl_utils import TLSContext, create_tls_context, generate_self_signed_cert


class TestSSLUtils(unittest.TestCase):
    def test_generate_self_signed_cert(self):
        """Test generating a self-signed certificate."""
        try:
            cert_path, key_path = generate_self_signed_cert()
            
            # Check that files exist
            self.assertTrue(os.path.exists(cert_path))
            self.assertTrue(os.path.exists(key_path))
            
            # Check file contents
            with open(cert_path, 'r') as f:
                cert_content = f.read()
                self.assertIn('BEGIN CERTIFICATE', cert_content)
            
            with open(key_path, 'r') as f:
                key_content = f.read()
                self.assertIn('BEGIN PRIVATE KEY', key_content)
        finally:
            # Clean up
            if os.path.exists(cert_path):
                os.remove(cert_path)
            if os.path.exists(key_path):
                os.remove(key_path)
            if os.path.exists(os.path.dirname(cert_path)):
                os.rmdir(os.path.dirname(cert_path))
    
    def test_tls_context(self):
        """Test creating a TLS context."""
        try:
            # Create a TLS context
            context = TLSContext()
            
            # Check that files were generated
            self.assertTrue(os.path.exists(context.cert_path))
            self.assertTrue(os.path.exists(context.key_path))
            
            # Create a server socket
            server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_sock.bind(('localhost', 0))
            server_sock.listen(1)
            port = server_sock.getsockname()[1]
            
            # Create wrapped server socket
            server_ssl_sock = context.wrap_socket(server_sock, server_side=True)
            
            # Create a client context
            client_context = TLSContext()
            
            # Create a client socket
            def client_thread():
                client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                client_sock.connect(('localhost', port))
                
                # Wrap client socket
                client_ssl_sock = client_context.wrap_socket(
                    client_sock,
                    server_hostname='localhost'
                )
                
                # Send data
                client_ssl_sock.send(b'Hello, server!')
                
                # Receive data
                data = client_ssl_sock.recv(1024)
                
                # Close socket
                client_ssl_sock.close()
                
                return data
            
            # Start client thread
            import threading
            client_data = None
            client_thread = threading.Thread(target=lambda: client_thread())
            client_thread.start()
            
            # Accept connection
            client_sock, addr = server_ssl_sock.accept()
            
            # Receive data
            data = client_sock.recv(1024)
            
            # Send response
            client_sock.send(b'Hello, client!')
            
            # Close socket
            client_sock.close()
            server_ssl_sock.close()
            
            # Join thread
            client_thread.join(timeout=5)
            
            # Check data
            self.assertEqual(data, b'Hello, server!')
        finally:
            # Clean up
            context.cleanup()
            client_context.cleanup()
    
    def test_context_manager(self):
        """Test using the context manager."""
        with create_tls_context() as context:
            self.assertIsInstance(context, TLSContext)
            self.assertTrue(os.path.exists(context.cert_path))
            self.assertTrue(os.path.exists(context.key_path))
        
        # Files should be cleaned up
        self.assertFalse(os.path.exists(context.cert_path))
        self.assertFalse(os.path.exists(context.key_path))


if __name__ == '__main__':
    unittest.main()