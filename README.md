# Arceus v0.5

## Introduction
Arceus is a distributed training library for PyTorch that enables efficient training of machine learning models across multiple devices. It automates device selection and simplifies the setup of distributed training environments.

## Features
- Automatic device detection (CUDA GPU > Apple Silicon MPS > CPU)
- Data parallel training
- Seamless integration with PyTorch
- Enhanced progress tracking with automatic metrics

## Installation

### Prerequisites
- Python 3.x
- PyTorch compatible with your environment

### Install
Using pip:
```bash
pip install -r requirements.txt
```

Using uv:
```bash
uv sync
```

## Usage

### Basic Example
```python
import torch
import torch.nn as nn
import torch.optim as optim
import arceus

# Initialize Arceus (automatically detects best device)
rank, world_size, args = arceus.init()

class NeuralNetwork(nn.Module):
    # Define your network architecture here
    pass

model = NeuralNetwork()
model = arceus.wrap(model)  # Move to best device + setup for distributed training

# Training setup
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(epochs):
    progress_bar = arceus.progress(dataloader, optimizer)
    for data, target in progress_bar:
        data, target = arceus.to_device(data), arceus.to_device(target)
        loss = criterion(model(data), target)
        loss.backward()
        optimizer.step()
        progress_bar.step(loss=loss)

arceus.finish()  # Clean up
```

### Command-Line Usage
#### Host a Session
```bash
python train.py --host
```

#### Join a Session
```bash
python train.py --join <session_id>
```

## macOS Distributed Training
Arceus configures Gloo for macOS to avoid common issues. For manual setup:

```python
import arceus

# Set up macOS-safe environment
arceus.setup_macos_env()

# Validate Gloo setup
arceus.validate_gloo()
```

### Troubleshooting
- Ensure devices are on the same network.
- Disable "client isolation" on routers.
- Allow Python in macOS Firewall settings.
- Try different ports if necessary.

## Contributing
We welcome contributions! Please read our [contributing guide](CONTRIBUTING.md) for more details.

## Support
For issues, please contact [support@example.com](mailto:support@example.com).

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
