import time
import uuid

from .networking import UDPBeacon
from .distributed import TrainingHost, TrainingJoiner
from .utils import banner, wait_for_sessions, pick_session, init_pytorch_distributed, BOLD, END

# Global state
_beacon = None
_rank = None
_world = None

def init(mode="auto", session=None, timeout=5):
    """
    Initialize distributed training
    
    mode: "host", "join", or "auto" 
    session: session ID to join (for join mode)
    timeout: how long to wait for session discovery
    
    Returns (rank, world_size)
    """
    global _beacon, _rank, _world
    
    # start discovery beacon
    _beacon = UDPBeacon("DISC", 0)
    time.sleep(0.3)  # give it a moment to start up
    
    # figure out what to do
    if mode == "auto":
        sessions = wait_for_sessions(_beacon, timeout)
        if sessions:
            mode = "join"
            session = pick_session(sessions)
        else:
            mode = "host"  # nobody else around, we'll be host
    elif mode == "join" and session is None:
        sessions = wait_for_sessions(_beacon, timeout)
        if not sessions:
            raise RuntimeError("No training sessions found on the network")
        session = pick_session(sessions)
    
    if mode == "host":
        # Generate random session ID
        session_id = uuid.uuid4().hex[:4].upper()
        host = TrainingHost(session_id)
        
        # Stop discovery beacon and start advertising our session
        _beacon.stop()
        _beacon = UDPBeacon(session_id, host.tcp_port)
        
        banner(f"\nSession ID: {BOLD}{session_id}{END} (share with peers)")
        print("Press Enter when everyone has joined with '--join " + session_id + "'")
        try:
            input()
        except (EOFError, KeyboardInterrupt):
            print("Starting anyway...")
        
        _world = host.start_training()
        _rank = 0
        
    else:  # join mode
        # Wait for the specific session to be discovered
        if session:
            print(f"Looking for session '{session}'...")
            deadline = time.time() + timeout
            
            while time.time() < deadline:
                sessions = _beacon.get_active_sessions()
                if session in sessions:
                    host_ip, host_port = sessions[session]
                    print(f"✓ Found session '{session}' at {host_ip}:{host_port}")
                    break
                time.sleep(0.3)
            else:
                raise RuntimeError(f"Session '{session}' not found after {timeout}s")
        else:
            host_ip, host_port = _beacon.get_active_sessions()[session]
        
        print(f"Connecting to session '{session}'...")
        joiner = TrainingJoiner(host_ip, host_port)
        joiner.connect_to_host()
        
        banner(f"Successfully joined session '{session}' - waiting for host to start training...")
        
        _world = joiner.wait_for_start()
        _rank = [uuid for uuid, _ in _world].index(joiner.my_id)
        
        print(f"Training starting as rank {_rank}/{len(_world)}")
    
    # Initialize PyTorch distributed
    init_pytorch_distributed(_world, _rank)
    
    return _rank, len(_world)

def wrap(model, show_graph=False):
    """add distributed gradient averaging hooks to model"""
    import torch
    import torch.fx
    
    if show_graph and _rank == 0:
        print("=== Model Graph ===")
        print(torch.fx.symbolic_trace(model).graph)
    
    # single process, nothing to do
    if len(_world) == 1:
        return model
    
    # hook to average gradients across all ranks
    def grad_hook(grad):
        import torch
        g_cpu = grad.detach().cpu()
        torch.distributed.all_reduce(g_cpu)
        g_cpu /= len(_world)
        return g_cpu.to(grad.device)
    
    for p in model.parameters():
        if p.requires_grad:
            p.register_hook(grad_hook)
    
    # sync initial weights so everyone starts the same
    import torch
    for p in model.parameters():
        torch.distributed.broadcast(p.data, src=0)
    
    return model

def progress(dataloader):
    """Create a progress bar for the dataloader"""
    from tqdm import tqdm
    return tqdm(dataloader, 
                position=_rank, 
                leave=False,
                bar_format=f"rank {_rank} {{l_bar}}{{bar}} {{n_fmt}}/{{total_fmt}}")

def finish():
    """Clean up distributed training"""
    import torch.distributed as dist
    dist.destroy_process_group()
    if _beacon:
        _beacon.stop()

def cli():
    """Parse command line arguments and initialize"""
    from .utils import parse_cli_args
    
    mode, session, args = parse_cli_args()
    rank, world_size = init(mode, session, args.timeout)
    return rank, world_size, args 