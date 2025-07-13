import argparse
import os
import time

# Environment variables
USE_AMP = os.getenv("ARCEUS_FP16", "0") == "1"
BUCKET_SIZE_KB = int(os.getenv("ARCEUS_BUCKET_KB", "0"))

# ANSI colors for pretty output
BOLD = "\033[1m"
CYAN = "\033[36m"
END = "\033[0m"

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
    # set up PyTorch distributed, retry if port conflicts
    import torch.distributed as dist
    from .networking import find_free_port
    
    master_ip, master_port = world[0][1]
    
    # try up to 10 times if ports are busy
    for i in range(10):
        try:
            os.environ["MASTER_ADDR"] = master_ip
            os.environ["MASTER_PORT"] = str(master_port)
            
            dist.init_process_group(
                "gloo",
                rank=rank,
                world_size=len(world),
                init_method=f"tcp://{master_ip}:{master_port}"
            )
            return  # success!
            
        except RuntimeError as e:
            if "EADDRINUSE" not in str(e):
                raise  # some other error
            # port busy, try another one
            master_port = find_free_port()
    
    raise RuntimeError("Couldn't init PyTorch distributed after 10 tries") 