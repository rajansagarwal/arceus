# arceus v0.5

distributed training across m-series macbooks for pytorch ml models

all you have to do is 

```python
...
import torch
import torch.nn as nn
import arceus

class NeuralNetwork(nn.Module):
    # your network architecture

...
model = NeuralNetwork()
model = arceus.wrap(model) # add this line
...
```

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