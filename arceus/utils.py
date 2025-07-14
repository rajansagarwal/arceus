import argparse
import os
import time

# Environment variables
USE_AMP = os.getenv("ARCEUS_FP16", "0") == "1"
BUCKET_SIZE_KB = int(os.getenv("ARCEUS_BUCKET_KB", "0"))

# ANSI colors for pretty output
BOLD = "\033[1m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
END = "\033[0m"

def detect_device():
    import torch
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
        device_info = f"CUDA ({device_name})"
        return device, device_info
    
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        device_info = "MPS (Apple Silicon)"
        return device, device_info
    
    else:
        device = torch.device("cpu")
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        device_info = f"CPU ({cpu_count} cores)"
        return device, device_info

def get_device_backend():
    import torch
    
    if torch.cuda.is_available():
        return "nccl"
    else:
        return "gloo"

def move_to_device(obj, device):
    try:
        return obj.to(device)
    except Exception as e:
        print(f"{YELLOW}Warning: Could not move to {device}, falling back to CPU{END}")
        return obj.to("cpu")

def print_device_info(device, device_info, rank=None):
    rank_str = f"[Rank {rank}] " if rank is not None else ""
    print(f"{GREEN}{rank_str}Using device: {device_info}{END}")

def banner(msg):
    print(BOLD + CYAN + msg + END)

def wait_for_sessions(beacon, timeout):
    """Wait for sessions to be discovered"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        sessions = beacon.get_active_sessions()
        if sessions:
            return sessions
        time.sleep(0.3)
    return {}

def pick_session(sessions):
    """Let user pick from available sessions"""
    print("\nAvailable sessions:")
    session_list = list(sessions.items())
    
    for i, (session_id, (ip, port)) in enumerate(session_list, 1):
        print(f"  {i}. {session_id} (host: {ip})")
    
    while True:
        try:
            choice = int(input("Select session: ")) - 1
            return session_list[choice][0]
        except (ValueError, IndexError):
            print("Invalid choice, try again")

def parse_cli_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--host", action="store_true", help="Start as host")
    group.add_argument("--join", nargs="?", const=True, default=False, help="Join session")
    parser.add_argument("--timeout", type=int, default=5, help="Discovery timeout")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs")
    parser.add_argument("--port", type=int, default=int(os.getenv("ARCEUS_MASTER_PORT", "29500")), help="Master port for PyTorch distributed (host)")
    
    args = parser.parse_args()
    
    # Figure out mode
    if args.host:
        mode = "host"
    elif args.join is not False:
        mode = "join"
    else:
        mode = "auto"
    
    # Figure out session
    if args.join is True:
        session = None  # will prompt user to pick
    else:
        session = args.join if args.join else None
    
    return mode, session, args

# Add a helper to dynamically pick the active network interface on macOS

def _pick_macos_iface() -> str:
    """Return the best UP, non-loopback network interface that can actually be bound to."""
    import subprocess, re, platform, socket
    if platform.system() != "Darwin":
        return "en0"  # sensible default on non-macOS (should not be called)
    
    import ipaddress

    try:
        ifconfig_out = subprocess.check_output(["ifconfig"]).decode()
        # Capture interface + its first IPv4 "inet " address (not inet6)
        pattern = re.compile(r"^(en\d+):.*?<UP,.*?>.*?(?:\n\s+.*)*?\n\s+inet (\d+\.\d+\.\d+\.\d+)", re.M | re.S)
        candidates = []
        
        for iface, ip in pattern.findall(ifconfig_out):
            try:
                ip_addr = ipaddress.ip_address(ip)
                
                # Skip loopback
                if ip_addr.is_loopback:
                    continue
                    
                # Skip carrier-grade NAT 100.64.0.0/10 which is not routable P2P
                if ip_addr >= ipaddress.ip_address("100.64.0.0") and ip_addr <= ipaddress.ip_address("100.127.255.255"):
                    continue
                
                # Test if we can actually bind to this interface
                try:
                    test_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    test_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    test_sock.bind((ip, 0))  # bind to any free port
                    test_sock.close()
                    
                    # If we got here, binding works
                    priority = 0
                    if iface == "en0":  # prefer en0 (usually Wi-Fi)
                        priority = 100
                    elif ip_addr.is_private:  # prefer private IPs
                        priority = 50
                    elif iface.startswith("en") and int(iface[2:]) < 4:  # prefer en0-en3
                        priority = 25
                    
                    candidates.append((priority, iface, ip))
                    
                except OSError:
                    # Can't bind to this interface, skip it
                    continue
                    
            except (ValueError, OSError):
                continue
        
        # Sort by priority (highest first) and return the best interface
        if candidates:
            candidates.sort(reverse=True)
            return candidates[0][1]
            
    except Exception:
        pass
    
    return "en0"  # fallback

def setup_macos_gloo_env():
    """Configure environment so that torch.distributed Gloo works reliably on macOS.

    The function follows the play-book agreed on by Apple & the community:
    1. Pick the first *UP* non-loopback interface (en0 on Wi-Fi, bridge100 for
       Internet-Sharing over USB-C, …) unless the user already exported
       GLOO_SOCKET_IFNAME.
    2. Resolve an IPv4 address for that interface and pin Gloo to it.
    3. Force IPv4 sockets only & disable IPv6 link-local picks.
    4. Keep a small, single-threaded socket pool for low-latency Wi-Fi links.
    """
    import subprocess, platform, socket

    if platform.system() != "Darwin":
        # Nothing to do on non-macOS hosts
        return None

    # Step 1 – pick interface
    iface = os.getenv("GLOO_SOCKET_IFNAME") or _pick_macos_iface()

    # Step 2 – resolve IPv4 for that interface
    try:
        ip_result = subprocess.run(["ipconfig", "getifaddr", iface],
                                   capture_output=True, text=True, check=True)
        ipaddr = ip_result.stdout.strip()
        
        # Validate that we can actually bind to this address
        try:
            test_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            test_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            test_sock.bind((ipaddr, 0))
            test_sock.close()
        except OSError as e:
            print(f"⚠️  Warning: Cannot bind to {iface} ({ipaddr}): {e}")
            # Fall back to alternative method
            try:
                ipaddr = socket.gethostbyname_ex(socket.gethostname())[2][0]
                print(f"   Falling back to hostname resolution: {ipaddr}")
            except Exception:
                ipaddr = "127.0.0.1"
                print(f"   Falling back to loopback: {ipaddr}")
        
    except Exception as e:
        print(f"⚠️  Warning: Could not get IP for {iface}: {e}")
        # Fallback: ask the OS for the default outbound IPv4
        try:
            ipaddr = socket.gethostbyname_ex(socket.gethostname())[2][0]
        except Exception:
            ipaddr = "127.0.0.1"

    # Step 3 – set / keep the golden env block
    os.environ.setdefault("GLOO_SOCKET_IFNAME", iface)
    os.environ.setdefault("GLOO_SOCKET_IFADDR", ipaddr)
    os.environ.setdefault("GLOO_SOCKET_FAMILY", "AF_INET")
    os.environ.setdefault("GLOO_SOCKET_DISABLE_IPV6", "1")
    os.environ.setdefault("GLOO_ALLOW_UNSECURED", "1")

    # Step 4 – small socket pool = lower overhead on Wi-Fi
    os.environ.setdefault("GLOO_SOCKET_NTHREADS", "1")
    os.environ.setdefault("GLOO_SOCKET_NSOCKS_PERTHREAD", "8")

    print(f"✓ macOS Gloo pinned to {iface} ({ipaddr})")
    return ipaddr

def validate_gloo_setup():
    """Quick validation of Gloo setup before running full distributed training"""
    import torch
    import torch.distributed as dist
    
    try:
        # Test single-rank Gloo initialization
        dist.init_process_group(
            backend="gloo",
            rank=0,
            world_size=1,
            init_method=f"tcp://{os.environ.get('GLOO_SOCKET_IFADDR', '127.0.0.1')}:29500",
        )
        print("✓ Gloo single-rank validation passed")
        dist.destroy_process_group()
        return True
    except Exception as e:
        print(f"✗ Gloo validation failed: {e}")
        return False

def init_pytorch_distributed(world, rank):
    import torch.distributed as dist
    from .networking import find_free_port
    
    master_ip, master_port = world[0][1]
    backend = get_device_backend()
    
    def _mask_ip(ip):
        """Mask IP address for privacy"""
        parts = ip.split('.')
        if len(parts) == 4:
            return f"{parts[0]}.{parts[1]}.xxx.xxx"
        return "xxx.xxx.xxx.xxx"
    
    # --- Harden Gloo against macOS firewall / IPv6 quirks ----------
    if backend == "gloo":
        import platform
        
        # Apply comprehensive macOS fixes
        if platform.system() == "Darwin":
            # Ensure macOS environment is set up
            setup_macos_gloo_env()
            
            # Environment already configured by setup_macos_gloo_env(); just log.
            iface  = os.environ.get("GLOO_SOCKET_IFNAME", "en0")
            ipaddr = os.environ.get("GLOO_SOCKET_IFADDR", world[rank][1][0])
            print(f"using Gloo on {iface} (IPv4), address: {_mask_ip(ipaddr)}, unsecured mode")
        else:
            # Non-macOS systems - minimal configuration
            os.environ.setdefault("GLOO_SOCKET_FAMILY", "AF_INET")
            ipaddr = world[rank][1][0]
            print(f"using Gloo (IPv4), address: {_mask_ip(ipaddr)}")
    
    # Additional debugging information
    if backend == "gloo" and os.getenv("ARCEUS_DEBUG"):
        print("Gloo environment variables:")
        for key in sorted(os.environ.keys()):
            if key.startswith("GLOO_"):
                print(f"  {key}: {os.environ[key]}")
    
    print(f"initializing distributed training...")
    print(f"  backend: {backend}")
    print(f"  master: {_mask_ip(master_ip)}:{master_port}")
    print(f"  rank: {rank}/{len(world)}")
    print(f"  world: {[(uuid[:8], _mask_ip(ip)) for uuid, (ip, port) in world]}")
    
    if rank == 0:
        print(f"  → rank 0 will LISTEN on port {master_port}")
    else:
        print(f"  → rank {rank} will CONNECT to {_mask_ip(master_ip)}:{master_port}")
    
    for attempt in range(3):  # Reduced attempts
        try:
            os.environ["MASTER_ADDR"] = master_ip
            os.environ["MASTER_PORT"] = str(master_port)
            
            print(f"  attempt {attempt + 1}/3: calling dist.init_process_group...")
            
            # 30-second window is kinder to flaky home Wi-Fi
            import datetime, os as _os
            timeout_secs = int(_os.getenv("ARCEUS_TIMEOUT", "30"))
            timeout = datetime.timedelta(seconds=timeout_secs)
            
            init_method = f"tcp://{master_ip}:{master_port}"
            print(f"  using init method: tcp://{_mask_ip(master_ip)}:{master_port}")
            
            dist.init_process_group(
                backend,
                rank=rank,
                world_size=len(world),
                init_method=init_method,
                timeout=timeout
            )
            print(f"✓ distributed training initialized successfully")
            return
            
        except RuntimeError as e:
            error_str = str(e).lower()
            if "timeout" in error_str:
                print(f"✗ attempt {attempt + 1} timed out after {timeout_secs}s")
                if attempt == 2:  # Last attempt
                    print("Common fixes:")
                    print("1. Ensure both devices are on the same WiFi network")
                    print("2. Check router settings - disable 'client isolation' or 'AP isolation'")  
                    print("3. Allow Python in macOS Firewall: System Settings → Network → Firewall")
                    print("4. Try different port: python train.py --host --port 8080")
                    print("5. For debugging: export ARCEUS_DEBUG=1")
                    raise RuntimeError("distributed training initialization timed out - check network connectivity")
                continue
            elif "unsupported gloo device" in error_str or "makedeviceforinterface" in error_str:
                print(f"✗ gloo interface issue - check network interface")
                print("Try: export GLOO_SOCKET_IFNAME=<your_interface>")
                raise
            elif "connection refused" in error_str or "connection failed" in error_str:
                print(f"✗ connection refused - firewall is likely blocking port {master_port}")
                print("Check macOS firewall settings and allow Python")
                raise
            elif "connectfullmesh" in error_str or "state_ != connecting" in error_str:
                print(f"✗ Gloo connection mesh failed - typical macOS firewall issue")
                print("This indicates the macOS stealth mode is still blocking connections")
                print("Try: sudo pfctl -d (temporarily disable firewall)")
                raise
            elif "fe80::" in error_str:
                print(f"✗ IPv6 link-local address detected - IPv6 not properly disabled")
                print("This should not happen with current configuration")
                raise
            elif "eaddrinuse" in error_str:
                print(f"  port {master_port} busy, trying another...")
                master_port = find_free_port()
                continue
            else:
                print(f"✗ distributed init failed: {e}")
                print("For debugging, set ARCEUS_DEBUG=1 and check Gloo environment variables")
                raise
    
    raise RuntimeError("couldn't initialize distributed training after 3 attempts") 