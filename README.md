# arceus v0.5

distributed training across consumer devices for pytorch ml models

<img width="1015" height="303" alt="image" src="https://github.com/user-attachments/assets/fc7ee527-29ab-4e73-a1c9-663240325b7e" />

currently only supports data parallel training with automatic device detection (CUDA GPU > Apple Silicon MPS > CPU). actively fixing model parallelism, not production ready yet. all you have to do is:
```python
...
import torch
import torch.nn as nn
import arceus

# Initialize arceus (automatically detects best device)
rank, world_size = arceus.init()

class NeuralNetwork(nn.Module):
    # your network architecture

...
model = NeuralNetwork()
model = arceus.wrap(model)  # automatically moves to best device + distributed training
...

# Move data to device easily
data = arceus.to_device(data)
target = arceus.to_device(target)
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
