import time
import uuid

from .networking import UDPBeacon
from .distributed import TrainingHost, TrainingJoiner
from .progress import MetricProgressBar
from .utils import (banner, wait_for_sessions, pick_session, init_pytorch_distributed, 
                   detect_device, print_device_info, move_to_device, BOLD, END)

# Global state
_beacon = None
_rank = None
_world = None
_device = None
_device_info = None

def init(mode="auto", session=None, timeout=5, port=None):
    global _beacon, _rank, _world, _device, _device_info
    
    _beacon = UDPBeacon("DISC", 0)
    time.sleep(0.3)
    
    # decide whether to host or join
    if mode == "auto":
        sessions = wait_for_sessions(_beacon, timeout)
        if sessions:
            mode = "join"
            session = pick_session(sessions)
        else:
            # No sessions found, run in single-process mode
            _beacon.stop()
            _world = [(str(uuid.uuid4()), ("localhost", 0))]
            _rank = 0
            
            _device, _device_info = detect_device()
            print_device_info(_device, _device_info, _rank)
            
            return _rank, len(_world)
    elif mode == "join" and session is None:
        sessions = wait_for_sessions(_beacon, timeout)
        if not sessions:
            raise RuntimeError("No training sessions found on the network")
        session = pick_session(sessions)
    
    if mode == "host":
        if port is None:
            import os
            port = int(os.getenv("ARCEUS_MASTER_PORT", "29500"))

        session_id = uuid.uuid4().hex[:4].upper()
        host = TrainingHost(session_id, port)
        
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
    
    # Only initialize distributed if we have multiple processes
    if len(_world) > 1:
        init_pytorch_distributed(_world, _rank)
    
    _device, _device_info = detect_device()
    print_device_info(_device, _device_info, _rank)
    
    return _rank, len(_world)

def _print_model_summary(model):
    """Print a clean, compact model summary"""
    from .utils import CYAN, BOLD, END
    
    print(f"{CYAN}{BOLD}model summary{END}")
    
    total_params = 0
    trainable_params = 0
    
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # leaf modules only
            param_count = sum(p.numel() for p in module.parameters())
            trainable_count = sum(p.numel() for p in module.parameters() if p.requires_grad)
            
            if param_count > 0:
                name = name if name else module.__class__.__name__
                if param_count >= 1_000_000:
                    param_str = f"{param_count/1_000_000:.1f}M"
                elif param_count >= 1_000:
                    param_str = f"{param_count/1_000:.1f}K" 
                else:
                    param_str = str(param_count)
                
                print(f"  {name:<20} {module.__class__.__name__:<15} {param_str:>8}")
                
            total_params += param_count
            trainable_params += trainable_count
    
    print(f"  {'-'*45}")
    
    if total_params >= 1_000_000:
        total_str = f"{total_params/1_000_000:.2f}M"
    elif total_params >= 1_000:
        total_str = f"{total_params/1_000:.1f}K"
    else:
        total_str = str(total_params)
    
    print(f"  {'total params':<36} {total_str:>8}")
    if trainable_params != total_params:
        print(f"  {'trainable params':<36} {trainable_params:>8}")

def wrap(model, show_graph=False, auto_device=True):
    import torch
    import torch.fx
    
    if show_graph and _rank == 0:
        _print_model_summary(model)
    
    if auto_device and _device is not None:
        model = move_to_device(model, _device)
    
    if len(_world) == 1:
        return model
    
    def grad_hook(grad):
        if _device.type == "cuda" and torch.distributed.get_backend() == "nccl":
            torch.distributed.all_reduce(grad)
            grad /= len(_world)
            return grad
        else:
            g_cpu = grad.detach().cpu()
            torch.distributed.all_reduce(g_cpu)
            g_cpu /= len(_world)
            return g_cpu.to(grad.device)
    
    for p in model.parameters():
        if p.requires_grad:
            p.register_hook(grad_hook)
    
    # sync initial weights
    print("synchronizing model parameters...")
    param_count = 0
    for p in model.parameters():
        param_count += 1
        if _device.type == "cuda" and torch.distributed.get_backend() == "nccl":
            torch.distributed.broadcast(p.data, src=0)
        else:
            p_cpu = p.data.cpu()
            torch.distributed.broadcast(p_cpu, src=0)
            p.data.copy_(p_cpu.to(p.device))
    print(f"✓ synchronized {param_count} parameter tensors")
    
    return model

def progress(dataloader, optimizer=None):
    return MetricProgressBar(dataloader, _rank, len(_world), _device, optimizer)

def get_device():
    return _device

def get_device_info():
    return _device_info

def to_device(obj):
    if _device is None:
        raise RuntimeError("arceus not initialized. Call arceus.init() first.")
    return move_to_device(obj, _device)

def finish():
    import torch.distributed as dist
    if dist.is_initialized():
        dist.destroy_process_group()
    if _beacon:
        _beacon.stop()

def cli():
    from .utils import parse_cli_args
    
    mode, session, args = parse_cli_args()
    rank, world_size = init(mode, session, args.timeout, args.port)
    return rank, world_size, args 

def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    return 0.0 