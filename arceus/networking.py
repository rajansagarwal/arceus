import contextlib
import json
import socket
import struct
import fcntl
import time
from threading import Thread

# network config stuff
DISCOVERY_PORT = 12346
MAGIC_HEADER = "FFTRAIN_DISC" 
BROADCAST_INTERVAL = 3.0  # seconds between broadcasts

# Utility function to get local IP address
# Tries multiple methods to ensure reliability

def get_local_ip():
    try:
        # use google DNS to figure out our local IP
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(("8.8.8.8", 80))
        ip = sock.getsockname()[0]
        sock.close()
        return ip
    except Exception:
        # fallback to ifconfig parsing on macOS
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            ip = socket.inet_ntoa(fcntl.ioctl(
                s.fileno(),
                0x8915,  # SIOCGIFADDR
                struct.pack('256s', b'en0'[:15])
            )[20:24])
            return ip
        except Exception as e:
            print(f"Failed to get local IP: {e}")
            return "127.0.0.1"

def get_broadcast_ip():
    local_ip = get_local_ip()
    ip_parts = local_ip.split('.')
    # Assuming a common subnet mask 255.255.255.0
    ip_parts[3] = '255'
    return '.'.join(ip_parts)

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
        
        # set up UDP socket for broadcasting
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if hasattr(socket, "SO_REUSEPORT"):  # not all systems have this
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self.sock.bind(("", DISCOVERY_PORT))
        
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
        dest = (get_broadcast_ip(), DISCOVERY_PORT)
        
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
        # Ensure threads are terminated
        print("Beacon stopped and resources cleaned up.")