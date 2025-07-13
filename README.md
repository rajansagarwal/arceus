# arceus v0.5

distributed training across m-series macbooks for pytorch ml models

all you have to do is 

```python
import arceus
...
model = arceus.wrap(model)
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