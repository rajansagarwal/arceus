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
    # use google DNS to figure out our local IPv4 address
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    finally:
        sock.close()

def get_local_ipv6():
    # use a public DNS IPv6 address to determine our local IPv6 address
    sock = socket.socket(socket.AF_INET6, socket.SOCK_DGRAM)
    try:
        sock.connect(("2001:4860:4860::8888", 80))
        return sock.getsockname()[0]
    finally:
        sock.close()

def get_broadcast_ip():
    # just replace last octet with 255 for IPv4
    parts = get_local_ip().split(".")
    parts[3] = "255"  
    return ".".join(parts)

def find_free_port():
    # let OS pick a free port
    with contextlib.closing(socket.socket()) as s:
        s.bind(("", 0))
        return s.getsockname()[1]

class UDPBeacon:
    """UDP beacon for finding other training sessions on the network"""
    
    def __init__(self, session_id, tcp_port):
        self.session_id = session_id
        self.tcp_port = tcp_port
        self.running = True
        self.peers = {}  # session_id -> (ip, port, timestamp)
        
        # set up UDP socket for broadcasting on IPv4
        self.sock_v4 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock_v4.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if hasattr(socket, "SO_REUSEPORT"):  # not all systems have this
            self.sock_v4.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        self.sock_v4.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self.sock_v4.bind(("", DISCOVERY_PORT))

        # set up UDP socket for broadcasting on IPv6
        self.sock_v6 = socket.socket(socket.AF_INET6, socket.SOCK_DGRAM)
        self.sock_v6.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if hasattr(socket, "SO_REUSEPORT"):  # not all systems have this
            self.sock_v6.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        self.sock_v6.bind(("", DISCOVERY_PORT))
        
        # start threads for tx/rx
        Thread(target=self._broadcast_loop, daemon=True).start()
        Thread(target=self._listen_loop, daemon=True).start()
    
    def _broadcast_loop(self):
        # keep broadcasting our session info
        msg_v4 = {
            "magic": MAGIC_HEADER,
            "session_id": self.session_id,
            "ip": get_local_ip(),
            "port": self.tcp_port
        }
        msg_v6 = {
            "magic": MAGIC_HEADER,
            "session_id": self.session_id,
            "ip": get_local_ipv6(),
            "port": self.tcp_port
        }
        packet_v4 = json.dumps(msg_v4).encode()
        packet_v6 = json.dumps(msg_v6).encode()
        dest_v4 = (get_broadcast_ip(), DISCOVERY_PORT)
        dest_v6 = ("ff02::1", DISCOVERY_PORT)  # link-local all-nodes multicast address for IPv6
        
        while self.running:
            try:
                self.sock_v4.sendto(packet_v4, dest_v4)
                self.sock_v6.sendto(packet_v6, dest_v6)
                time.sleep(BROADCAST_INTERVAL)
            except OSError:
                break  # socket probably closed
    
    def _listen_loop(self):
        # listen for broadcasts from other sessions
        while self.running:
            try:
                data_v4, addr_v4 = self.sock_v4.recvfrom(1024)
                info_v4 = json.loads(data_v4.decode())
                if info_v4.get("magic") == MAGIC_HEADER:
                    self.peers[info_v4["session_id"]] = (
                        info_v4["ip"], 
                        info_v4["port"], 
                        time.time()
                    )

                data_v6, addr_v6 = self.sock_v6.recvfrom(1024)
                info_v6 = json.loads(data_v6.decode())
                if info_v6.get("magic") == MAGIC_HEADER:
                    self.peers[info_v6["session_id"]] = (
                        info_v6["ip"], 
                        info_v6["port"], 
                        time.time()
                    )
                    
            except OSError:
                break
    
    def get_active_sessions(self):
        # return sessions we've heard from recently
        now = time.time()
        active = {}
        
        for session_id, (ip, port, ts) in self.peers.items():
            # ensure we only return sessions that are recent and valid
            if session_id != "DISC" and port > 0 and (now - ts) < 10:
                active[session_id] = (ip, port)
        
        return active
    
    def stop(self):
        self.running = False
        self.sock_v4.close()
        self.sock_v6.close()