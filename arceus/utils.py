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
    
    # --- Mac-friendly Gloo setup ---------------------------------
    os.environ["GLOO_SOCKET_FAMILY"] = "AF_INET"     # IPv4 only
    os.environ["GLOO_SOCKET_IFNAME"] = "en0"         # bind to Wi-Fi
    os.environ["GLOO_ALLOW_UNSECURED"] = "1"         # skip stealth-mode RST
    print("using Gloo on interface en0 (IPv4), unsecured mode")
    
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
            
            # Reduced timeout for faster feedback
            import datetime
            timeout = datetime.timedelta(seconds=10)
            
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
                print(f"✗ attempt {attempt + 1} timed out after 10s")
                if attempt == 2:  # Last attempt
                    print("Common fixes:")
                    print("1. Ensure both devices are on the same WiFi network")
                    print("2. Check router settings - disable 'client isolation' or 'AP isolation'")  
                    print("3. Try different port: python train.py --host --port 8080")
                    raise RuntimeError("distributed training initialization timed out - check network connectivity")
                continue
            elif "unsupported gloo device" in error_str or "makedeviceforinterface" in error_str:
                print(f"✗ gloo interface issue - check network interface")
                raise
            elif "connection refused" in error_str or "connection failed" in error_str:
                print(f"✗ connection refused - firewall is likely blocking port {master_port}")
                raise
            elif "eaddrinuse" in error_str:
                print(f"  port {master_port} busy, trying another...")
                master_port = find_free_port()
                continue
            else:
                print(f"✗ distributed init failed: {e}")
                raise
    
    raise RuntimeError("couldn't initialize distributed training after 3 attempts") 