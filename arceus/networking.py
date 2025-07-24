import contextlib
import json
import socket
import time
from threading import Thread

# network config stuff
DISCOVERY_PORT = 12346
MAGIC_HEADER = "FFTRAIN_DISC" 
BROADCAST_INTERVAL = 3.0  # seconds between broadcasts

def get_local_ip():
    # get the local IPv6 address
    sock = socket.socket(socket.AF_INET6, socket.SOCK_DGRAM)
    try:
        sock.connect(("2001:4860:4860::8888", 80))  # Google's public DNS IPv6
        return sock.getsockname()[0]
    finally:
        sock.close()

# Broadcast logic does not have a direct IPv6 equivalent

def find_free_port():
    # let OS pick a free port
    with contextlib.closing(socket.socket(socket.AF_INET6, socket.SOCK_DGRAM)) as s:
        s.bind(("::", 0))
        return s.getsockname()[1]

class UDPBeacon:
    """UDP beacon for finding other training sessions on the network"""
    
    def __init__(self, session_id, tcp_port):
        self.session_id = session_id
        self.tcp_port = tcp_port
        self.running = True
        self.peers = {}  # session_id -> (ip, port, timestamp)
        
        # set up UDP socket for broadcasting
        self.sock = socket.socket(socket.AF_INET6, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if hasattr(socket, "SO_REUSEPORT"):  # not all systems have this
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        # Broadcasting in IPv6 is different; often multicast is used instead
        self.sock.bind(("::", DISCOVERY_PORT))
        
        # start threads for tx/rx
        Thread(target=self._broadcast_loop, daemon=True).start()
        Thread(target=self._listen_loop, daemon=True).start()
    
    def _broadcast_loop(self):
        # keep broadcasting our session info
        msg = {
            "magic": MAGIC_HEADER,
            "session_id": self.session_id,
            "ip": get_local_ip(),
            "port": self.tcp_port
        }
        packet = json.dumps(msg).encode()
        dest = ("ff02::1", DISCOVERY_PORT)  # Multicast address for all nodes
        
        while self.running:
            try:
                self.sock.sendto(packet, dest)
                time.sleep(BROADCAST_INTERVAL)
            except OSError:
                break  # socket probably closed
    
    def _listen_loop(self):
        # listen for broadcasts from other sessions
        while self.running:
            try:
                data, addr = self.sock.recvfrom(1024)
                info = json.loads(data.decode())
                
                if info.get("magic") != MAGIC_HEADER:
                    continue  # not one of ours
                    
                self.peers[info["session_id"]] = (
                    info["ip"], 
                    info["port"], 
                    time.time()
                )
            except OSError:
                break
    
    def get_active_sessions(self):
        # return sessions we've heard from recently
        now = time.time()
        active = {}
        
        for session_id, (ip, port, ts) in self.peers.items():
            # skip our own discovery beacon and expired ones
            if session_id != "DISC" and port > 0 and (now - ts) < 10:
                active[session_id] = (ip, port)
        
        return active
    
    def stop(self):
        self.running = False
        self.sock.close()