import contextlib
import json
import os
import socket
import time
from threading import Thread

# network config stuff
DISCOVERY_PORT = 12346
MAGIC_HEADER = "FFTRAIN_DISC" 
BROADCAST_INTERVAL = 3.0  # seconds between broadcasts

def get_local_ip():
    # use google DNS to figure out our local IP
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    finally:
        sock.close()

def get_broadcast_ip():
    """Get broadcast IP addresses for all available network interfaces.
    
    For cross-device communication, we need to broadcast on all possible
    network segments, not just assume a /24 subnet.
    """
    import netifaces
    import ipaddress
    
    broadcast_ips = []
    
    try:
        for interface in netifaces.interfaces():
            try:
                addrs = netifaces.ifaddresses(interface)
                if netifaces.AF_INET in addrs:
                    for addr_info in addrs[netifaces.AF_INET]:
                        if 'broadcast' in addr_info and 'addr' in addr_info:
                            broadcast_ip = addr_info['broadcast']
                            local_ip = addr_info['addr']
                            
                            try:
                                ip_obj = ipaddress.IPv4Address(local_ip)
                                if not ip_obj.is_loopback and not ip_obj.is_link_local:
                                    broadcast_ips.append(broadcast_ip)
                            except ValueError:
                                continue
            except (KeyError, ValueError):
                continue
                
    except ImportError:
        pass
    
    if not broadcast_ips:
        try:
            parts = get_local_ip().split(".")
            parts[3] = "255"  
            broadcast_ips.append(".".join(parts))
        except:
            broadcast_ips.append("255.255.255.255")
    
    return broadcast_ips[0]

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
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
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
        
        broadcast_ips = self._get_all_broadcast_ips()
        
        while self.running:
            try:
                for broadcast_ip in broadcast_ips:
                    try:
                        dest = (broadcast_ip, DISCOVERY_PORT)
                        self.sock.sendto(packet, dest)
                    except OSError as e:
                        if os.getenv("ARCEUS_DEBUG"):
                            print(f"Failed to broadcast to {broadcast_ip}: {e}")
                time.sleep(BROADCAST_INTERVAL)
            except OSError:
                break  # socket probably closed
    
    def _get_all_broadcast_ips(self):
        """Get all broadcast IP addresses for cross-device discovery."""
        import netifaces
        import ipaddress
        
        broadcast_ips = []
        
        try:
            for interface in netifaces.interfaces():
                try:
                    addrs = netifaces.ifaddresses(interface)
                    if netifaces.AF_INET in addrs:
                        for addr_info in addrs[netifaces.AF_INET]:
                            if 'broadcast' in addr_info and 'addr' in addr_info:
                                broadcast_ip = addr_info['broadcast']
                                local_ip = addr_info['addr']
                                
                                try:
                                    ip_obj = ipaddress.IPv4Address(local_ip)
                                    if not ip_obj.is_loopback and not ip_obj.is_link_local:
                                        broadcast_ips.append(broadcast_ip)
                                except ValueError:
                                    continue
                except (KeyError, ValueError):
                    continue
                    
        except ImportError:
            pass
        
        if not broadcast_ips:
            try:
                parts = get_local_ip().split(".")
                parts[3] = "255"  
                broadcast_ips.append(".".join(parts))
            except:
                broadcast_ips.append("255.255.255.255")
        
        return broadcast_ips
    
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