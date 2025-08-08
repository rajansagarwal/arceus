# Arceus

Distributed training framework for efficient multi-device training.

## Overview

Arceus is a lightweight distributed training framework built on top of PyTorch's distributed package. It provides:

- Simple API for distributed training
- Auto-discovery of training sessions on local network
- Efficient synchronization of model parameters
- Progress tracking and metrics visualization
- Cross-platform support (including Apple Silicon)

## Installation

```bash
pip install -e .
```

## Usage

Basic usage example:

```python
import arceus
import torch

# Initialize arceus with CLI arguments
rank, world_size = arceus.cli()

# Create and distribute your model
model = YourModel()
model = arceus.wrap(model)

# Train as usual, arceus handles the distributed part
for epoch in range(10):
    for batch in dataloader:
        # your training code
        pass
```

## Cross-Macbook Communication

For reliable cross-Macbook communication, Arceus now forces IPv4 usage throughout the codebase. This resolves issues with IPv6 link-local addresses and improves connection reliability.

Key features:
- Explicit IPv4 socket family (AF_INET) for all sockets
- Improved network interface detection on macOS
- IPv6 disabled by default for all PyTorch distributed communication
- Better handling of network interface selection
- Enhanced error messages for network-related issues

### Testing IPv4 Configuration

You can validate your network configuration using the included test script:

```bash
python test_ipv4.py
```

This will check:
- IPv4 address detection
- Socket binding with IPv4
- Network interface status

### Troubleshooting

If you encounter connection issues:

1. Ensure both devices are on the same network
2. Check your firewall settings (allow Python/PyTorch)
3. Try with explicit interface: `export GLOO_SOCKET_IFNAME=en0`
4. Set `export ARCEUS_DEBUG=1` for verbose logging
5. If still having issues, try setting `export ARCEUS_TIMEOUT=60` for longer timeout

## Command Line Options

- `--host`: Start as session host
- `--join [SESSION_ID]`: Join existing session
- `--timeout SECONDS`: Discovery timeout (default: 5s)
- `--epochs N`: Number of training epochs
- `--port PORT`: Port for PyTorch distributed (host mode)