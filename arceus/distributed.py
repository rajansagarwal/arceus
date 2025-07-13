import json
import select
import socket
import uuid
from threading import Thread
from typing import List, Tuple

from .networking import get_local_ip, find_free_port

class TrainingHost:
    """Host side of distributed training setup"""
    
    def __init__(self, session_id, master_port):
        self.session_id = session_id
        self.tcp_port = find_free_port()
        self.master_port = master_port  # fixed port for PyTorch distributed
        self.host_uuid = str(uuid.uuid4())
        
        # TCP server to accept joiner connections
        self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_sock.bind(("", self.tcp_port))
        self.server_sock.listen(8)  # max 8 pending connections
        
        self.clients = {}  # uuid -> socket
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
                # receive client_id and their IP address
                data = client_sock.recv(256).decode()
                parts = data.split(':')
                client_id = parts[0]
                client_ip = parts[1] if len(parts) > 1 else addr[0]
                
                self.clients[client_id] = (client_sock, client_ip)
                print(f"✅ Peer joined: {client_id[:8]}...")
                
            except OSError:
                break  # probably shutting down
    
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
            except:
                pass  # client might have disconnected already
        
        self.server_sock.close()
        return world

class TrainingJoiner:
    """Client side for joining a training session"""
    
    def __init__(self, host_ip, host_port):
        self.host_ip = host_ip
        self.host_port = host_port
        self.my_id = str(uuid.uuid4())
        self.sock = None
    
    def connect_to_host(self):
        # connect to host and register ourselves
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.host_ip, self.host_port))
        
        # send our ID and IP address
        from .networking import get_local_ip
        my_ip = get_local_ip()
        data = f"{self.my_id}:{my_ip}"
        self.sock.send(data.encode())
        # connected but don't wait for start yet
    
    def wait_for_start(self):
        # wait for host to tell us to start training
        data = self.sock.recv(4096)
        msg = json.loads(data.decode())
        self.sock.close()
        
        world = msg["world"]
        
        # make sure we're in the world list somehow
        if all(peer_id != self.my_id for peer_id, _ in world):
            world.append((self.my_id, (self.host_ip, world[0][1][1])))
        
        return world  # don't sort! host already sent it in the right order 