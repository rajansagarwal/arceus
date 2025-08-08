import json
import select
import socket
import ssl
import uuid
from threading import Thread
from typing import List, Tuple, Optional

from .networking import get_local_ip, find_free_port
from .ssl_utils import TLSContext, create_tls_context
from .utils import USE_TLS, TLS_VERIFY

class TrainingHost:
    """Host side of distributed training setup"""
    
    def __init__(self, session_id, master_port, use_tls=USE_TLS, 
                 cert_path=None, key_path=None, verify_mode=ssl.CERT_NONE):
        self.session_id = session_id
        self.tcp_port = find_free_port()
        self.master_port = master_port  # fixed port for PyTorch distributed
        self.host_uuid = str(uuid.uuid4())
        self.use_tls = use_tls
        self.tls_context = None
        
        # Set up TLS context if enabled
        if self.use_tls:
            verify_mode = ssl.CERT_REQUIRED if TLS_VERIFY else ssl.CERT_NONE
            self.tls_context = TLSContext(
                cert_path=cert_path,
                key_path=key_path,
                verify_mode=verify_mode,
                server_side=True
            )
        
        # TCP server to accept joiner connections
        self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_sock.bind(("", self.tcp_port))
        self.server_sock.listen(8)  # max 8 pending connections
        
        self.clients = {}  # uuid -> (socket, client_ip)
        self.accepting = True
        
        # start accepting in background thread
        Thread(target=self._accept_loop, daemon=True).start()
    
    def _accept_loop(self):
        # accept incoming joiner connections
        while self.accepting:
            try:
                ready, _, _ = select.select([self.server_sock], [], [], 1)
                if not ready:
                    continue
                
                client_sock, addr = self.server_sock.accept()
                
                # Wrap socket with TLS if enabled
                if self.use_tls and self.tls_context:
                    try:
                        client_sock = self.tls_context.wrap_socket(client_sock, server_side=True)
                        print(f"🔒 TLS connection established with {addr[0]}")
                    except ssl.SSLError as e:
                        print(f"⚠️ TLS handshake failed with {addr[0]}: {e}")
                        client_sock.close()
                        continue
                
                # Receive client_id and their IP address
                data = client_sock.recv(256).decode()
                parts = data.split(':')
                client_id = parts[0]
                client_ip = parts[1] if len(parts) > 1 else addr[0]
                
                self.clients[client_id] = (client_sock, client_ip)
                print(f"✅ Peer joined: {client_id[:8]}...")
                
            except (OSError, ssl.SSLError) as e:
                print(f"⚠️ Connection error: {e}")
                if not self.accepting:
                    break  # probably shutting down
                continue
    
    def start_training(self):
        # send start signal to everyone
        self.accepting = False
        
        # build world list - host is always rank 0
        world = [(self.host_uuid, (get_local_ip(), self.master_port))]
        
        # add all the joiners with their actual IP addresses
        for client_id in sorted(self.clients.keys()):
            sock, client_ip = self.clients[client_id]
            world.append((client_id, (client_ip, self.master_port)))
        
        # tell everyone to start
        msg = json.dumps({
            "start": True,
            "world": world
        }).encode()
        
        for client_id, (sock, _) in self.clients.items():
            try:
                sock.sendall(msg)
                sock.close()
            except Exception as e:
                print(f"⚠️ Failed to send start signal to {client_id[:8]}: {e}")
                # client might have disconnected already
        
        self.server_sock.close()
        
        # Clean up TLS context if it exists
        if self.tls_context:
            self.tls_context.cleanup()
        
        return world

class TrainingJoiner:
    """Client side for joining a training session"""
    
    def __init__(self, host_ip, host_port, use_tls=False, cert_path=None, key_path=None):
        self.host_ip = host_ip
        self.host_port = host_port
        self.my_id = str(uuid.uuid4())
        self.use_tls = use_tls
        self.tls_context = None
        self.sock = None
        
        # Set up TLS context if enabled
        if self.use_tls:
            self.tls_context = TLSContext(
                cert_path=cert_path,
                key_path=key_path,
                verify_mode=ssl.CERT_NONE,  # Accept self-signed certificates
                server_side=False
            )
    
    def connect_to_host(self):
        # connect to host and register ourselves
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        try:
            self.sock.connect((self.host_ip, self.host_port))
            
            # Wrap socket with TLS if enabled
            if self.use_tls and self.tls_context:
                try:
                    self.sock = self.tls_context.wrap_socket(
                        self.sock,
                        server_hostname=self.host_ip  # For SNI
                    )
                    print(f"🔒 TLS connection established with host")
                except ssl.SSLError as e:
                    print(f"⚠️ TLS handshake failed: {e}")
                    raise ConnectionError(f"TLS handshake failed: {e}")
            
            # send our ID and IP address
            from .networking import get_local_ip
            my_ip = get_local_ip()
            data = f"{self.my_id}:{my_ip}"
            self.sock.send(data.encode())
            # connected but don't wait for start yet
            
        except Exception as e:
            if self.sock:
                self.sock.close()
                self.sock = None
            raise ConnectionError(f"Failed to connect to host: {e}")
    
    def wait_for_start(self):
        # wait for host to tell us to start training
        try:
            data = self.sock.recv(4096)
            msg = json.loads(data.decode())
        except Exception as e:
            print(f"⚠️ Failed to receive start signal: {e}")
            raise
        finally:
            # Clean up
            if self.sock:
                self.sock.close()
                self.sock = None
            
            # Clean up TLS context if it exists
            if self.tls_context:
                self.tls_context.cleanup()
        
        world = msg["world"]
        
        # make sure we're in the world list somehow
        if all(peer_id != self.my_id for peer_id, _ in world):
            world.append((self.my_id, (self.host_ip, world[0][1][1])))
        
        return world  # don't sort! host already sent it in the right order 