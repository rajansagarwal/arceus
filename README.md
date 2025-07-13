# arceus v0.5

distributed training across multiple devices for pytorch ml models

supports data parallel training with automatic device detection (CUDA GPU > Apple Silicon MPS > CPU). currently fixing model parallelism, not production ready yet. here's how it works:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import arceus

# initialize arceus (automatically detects best device)
rank, world_size, args = arceus.init()

class NeuralNetwork(nn.Module):
    # your network architecture
    pass

model = NeuralNetwork()
model = arceus.wrap(model)  # automatically moves to best device + distributed training

optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

# Enhanced progress bar with automatic metrics
for epoch in range(epochs):
    progress_bar = arceus.progress(dataloader, optimizer)
    
    for data, target in progress_bar:
        data, target = arceus.to_device(data), arceus.to_device(target)
        
        # ... training step ...
        loss = criterion(model(data), target)
        loss.backward()
        optimizer.step()
        
        # metrics captured automatically! (host sees aggregated, joiners see individual)
        progress_bar.step(loss=loss)  # pass any metrics: loss=x, accuracy=y, etc.
            
...
arceus.finish()  # clean up distributed training
```

device selection is automatic - arceus chooses the best available device and configures the appropriate distributed backend.

## Usage

### installation

```bash
# if using uv 
uv sync
# if using pip
pip install -r requirements.txt
```

### host

```
python train.py --host
```

### join

```
python train.py --join <session_id>
```

## macOS distributed training

arceus automatically configures Gloo for macOS to avoid common firewall/IPv6 issues. for manual setup:

```python
import arceus

# set up macOS-safe environment (optional - done automatically)
arceus.setup_macos_env()

# validate Gloo setup (optional - for debugging)
arceus.validate_gloo()
```

### troubleshooting

if you get `Connection reset by peer` or `state_ != CONNECTING` errors:

1. ensure both devices are on the same WiFi network
2. check router settings - disable "client isolation" or "AP isolation"  
3. allow Python in macOS Firewall: System Settings → Network → Firewall
4. try different port: `python train.py --host --port 8080`

the library automatically sets these environment variables for macOS:
- `GLOO_SOCKET_FAMILY=AF_INET` (IPv4 only)
- `GLOO_SOCKET_DISABLE_IPV6=1` (block fe80::* picks)
- `GLOO_SOCKET_IFNAME=en0` (bind to Wi-Fi)
- `GLOO_SOCKET_IFADDR=<wifi_ip>` (pin to actual IP)
- `GLOO_ALLOW_UNSECURED=1` (skip stealth-mode RST)
