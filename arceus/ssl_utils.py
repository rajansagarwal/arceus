import os
import ssl
import socket
import tempfile
import contextlib
from datetime import datetime, timedelta
from typing import Tuple, Optional, Dict, Any

# Default certificate validity period (in days)
DEFAULT_CERT_VALIDITY_DAYS = 30

def generate_self_signed_cert(
    common_name: str = None,
    validity_days: int = DEFAULT_CERT_VALIDITY_DAYS
) -> Tuple[str, str]:
    """Generate a self-signed certificate and private key.
    
    Args:
        common_name: Common name for the certificate (defaults to hostname)
        validity_days: Number of days the certificate will be valid
        
    Returns:
        Tuple[str, str]: (certificate_path, key_path)
    """
    try:
        from cryptography import x509
        from cryptography.x509.oid import NameOID
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.hazmat.backends import default_backend
    except ImportError:
        raise ImportError(
            "cryptography package is required for TLS support. "
            "Install it with: pip install cryptography"
        )
    
    # Use hostname if common_name not provided
    if common_name is None:
        common_name = socket.gethostname()
    
    # Create a temporary directory to store the cert and key
    temp_dir = tempfile.mkdtemp(prefix="arceus_ssl_")
    cert_path = os.path.join(temp_dir, "cert.pem")
    key_path = os.path.join(temp_dir, "key.pem")
    
    # Generate private key
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
        backend=default_backend()
    )
    
    # Generate self-signed certificate
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, common_name),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Arceus Training"),
        x509.NameAttribute(NameOID.ORGANIZATIONAL_UNIT_NAME, "Distributed Training"),
    ])
    
    now = datetime.utcnow()
    cert = x509.CertificateBuilder().subject_name(
        subject
    ).issuer_name(
        issuer
    ).public_key(
        private_key.public_key()
    ).serial_number(
        x509.random_serial_number()
    ).not_valid_before(
        now
    ).not_valid_after(
        now + timedelta(days=validity_days)
    ).add_extension(
        x509.SubjectAlternativeName([
            x509.DNSName(common_name),
            x509.DNSName(socket.gethostname()),
            x509.DNSName("localhost"),
        ]),
        critical=False,
    ).sign(private_key, hashes.SHA256(), default_backend())
    
    # Write certificate and key to files
    with open(cert_path, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))
    
    with open(key_path, "wb") as f:
        f.write(private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        ))
    
    return cert_path, key_path

class TLSContext:
    """Manages SSL context for secure connections."""
    
    def __init__(
        self,
        cert_path: Optional[str] = None,
        key_path: Optional[str] = None,
        verify_mode: ssl.VerifyMode = ssl.CERT_NONE,
        server_side: bool = False
    ):
        """Initialize a TLS context.
        
        Args:
            cert_path: Path to certificate file
            key_path: Path to private key file
            verify_mode: SSL verification mode
            server_side: Whether this is a server-side context
        """
        self.cert_path = cert_path
        self.key_path = key_path
        self.verify_mode = verify_mode
        self.server_side = server_side
        self._temp_dir = None
        self._context = None
        
        # Generate certificate if not provided
        if not cert_path or not key_path:
            self._temp_dir = tempfile.mkdtemp(prefix="arceus_ssl_")
            self.cert_path, self.key_path = generate_self_signed_cert()
        
        # Create SSL context
        self._create_context()
    
    def _create_context(self):
        """Create an SSL context with the certificate and key."""
        context = ssl.create_default_context(
            ssl.Purpose.CLIENT_AUTH if self.server_side else ssl.Purpose.SERVER_AUTH
        )
        
        # Load certificate and key
        context.load_cert_chain(certfile=self.cert_path, keyfile=self.key_path)
        
        # Set verification mode
        context.verify_mode = self.verify_mode
        
        # Server-side specific configuration
        if self.server_side:
            context.check_hostname = False
        
        self._context = context
    
    def wrap_socket(self, sock, **kwargs):
        """Wrap a socket with TLS."""
        # Ensure the context exists
        if self._context is None:
            self._create_context()
        
        # Default server_side to the value provided during initialization
        kwargs.setdefault('server_side', self.server_side)
        
        return self._context.wrap_socket(sock, **kwargs)
    
    def cleanup(self):
        """Clean up temporary files."""
        if self._temp_dir and os.path.exists(self._temp_dir):
            import shutil
            shutil.rmtree(self._temp_dir)
            self._temp_dir = None

@contextlib.contextmanager
def create_tls_context(server_side=False, **kwargs):
    """Context manager for creating and cleaning up a TLS context.
    
    Args:
        server_side: Whether this is a server-side context
        **kwargs: Additional arguments for TLSContext
        
    Yields:
        TLSContext: The TLS context
    """
    context = TLSContext(server_side=server_side, **kwargs)
    try:
        yield context
    finally:
        context.cleanup()

def wrap_socket_tls(sock, server_side=False, **kwargs):
    """Convenience function to wrap a socket with TLS.
    
    Args:
        sock: The socket to wrap
        server_side: Whether this is a server-side socket
        **kwargs: Additional arguments for TLSContext
        
    Returns:
        ssl.SSLSocket: The wrapped socket
    """
    with create_tls_context(server_side=server_side, **kwargs) as context:
        return context.wrap_socket(sock, server_side=server_side)