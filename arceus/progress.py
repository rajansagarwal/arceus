import torch
from tqdm import tqdm


class MetricProgressBar:
    def __init__(self, dataloader, rank, world_size, device=None, optimizer=None):
        self.rank = rank
        self.world_size = world_size
        self.is_host = (rank == 0)
        self.device = device
        self.optimizer = optimizer
        self.metrics = {}
        self.batch_count = 0
        
        if self.is_host:
            bar_format = f"host (aggregated) {{l_bar}}{{bar}} {{n_fmt}}/{{total_fmt}} | {{postfix}}"
        else:
            bar_format = f"rank {rank} {{l_bar}}{{bar}} {{n_fmt}}/{{total_fmt}} | {{postfix}}"
        
        self.pbar = tqdm(dataloader, 
                        position=rank, 
                        leave=False,
                        bar_format=bar_format)
    
    def step(self, **kwargs):
        """Update metrics. Pass any metrics you want to track!"""
        self.batch_count += 1
        
        # auto-capture learning rate if optimizer provided
        if self.optimizer is not None and 'lr' not in kwargs:
            lr = self._get_lr()
            if lr is not None:
                kwargs['lr'] = lr
        
        # store all metrics
        for key, value in kwargs.items():
            if torch.is_tensor(value):
                value = value.item()
            self.metrics[key] = value
        
        # only update display every few batches to avoid overhead
        # or if new metrics provided
        if self.batch_count % 5 == 0 or kwargs:
            self._update_display(kwargs)
    
    def _get_lr(self):
        """Get current learning rate from optimizer"""
        if self.optimizer and hasattr(self.optimizer, 'param_groups'):
            return self.optimizer.param_groups[0]['lr']
        return None
    
    def _update_display(self, latest_kwargs):
        if self.world_size > 1 and torch.distributed.is_initialized():
            try:
                display_metrics = self._aggregate_metrics(latest_kwargs)
                if self.is_host:
                    metric_str = self._format_metrics(display_metrics)
                    self.pbar.set_postfix_str(metric_str)
            except Exception:
                metric_str = self._format_metrics(self.metrics)
                self.pbar.set_postfix_str(metric_str)
        else:
            metric_str = self._format_metrics(self.metrics)
            self.pbar.set_postfix_str(metric_str)
    
    def update_metrics(self, **kwargs):
        """Legacy method for manual metric updates"""
        self.step(**kwargs)
    
    def _aggregate_metrics(self, local_metrics):
        aggregated = {}
        
        for key, value in local_metrics.items():
            if torch.is_tensor(value):
                value = value.item()
            
            # use the same device strategy as the distributed backend
            try:
                backend = torch.distributed.get_backend()
                if backend == "nccl" and self.device and self.device.type == "cuda":
                    tensor_val = torch.tensor(float(value), device=self.device)
                else:
                    tensor_val = torch.tensor(float(value))
            except Exception:
                # fall back to CPU tensor if backend check fails
                tensor_val = torch.tensor(float(value))
            
            # all ranks must participate in all_reduce
            torch.distributed.all_reduce(tensor_val, op=torch.distributed.ReduceOp.SUM)
            aggregated[key] = tensor_val.item() / self.world_size
        
        return aggregated
    
    def _format_metrics(self, metrics):
        formatted = []
        for key, value in metrics.items():
            if isinstance(value, float):
                if abs(value) >= 1e-3:
                    formatted.append(f"{key}: {value:.4f}")
                else:
                    formatted.append(f"{key}: {value:.2e}")
            else:
                formatted.append(f"{key}: {value}")
        return " | ".join(formatted)
    
    def __iter__(self):
        return iter(self.pbar)
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.pbar.close()
    
    def close(self):
        self.pbar.close() 