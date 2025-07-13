import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import arceus

# initialize arceus with CLI args
rank, world_size, args = arceus.cli()

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )
    
    def forward(self, x):
        return self.layers(x)

class DummyDataset(Dataset):
    def __len__(self):
        return 60000
    
    def __getitem__(self, idx):
        return torch.randn(1, 28, 28), torch.randint(0, 10, (1,)).item()

# setup model and distributed wrapper
model = Net()
model = arceus.wrap(model, show_graph=(rank == 0))

# training setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())
dataloader = DataLoader(DummyDataset(), batch_size=128, shuffle=True)

# training loop
for epoch in range(args.epochs):
    progress_bar = arceus.progress(dataloader, optimizer)
    
    for data, target in progress_bar:
        data, target = arceus.to_device(data), arceus.to_device(target)
        
        optimizer.zero_grad()
        output = model(data)
        
        # amp if enabled
        if arceus._USE_AMP:
            device = arceus.get_device()
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                loss = criterion(output, target)
        else:
            loss = criterion(output, target)
        
        loss.backward()
        optimizer.step()
        
        # metrics are captured automatically! pass whatever you want to track
        progress_bar.step(loss=loss)
    
    # only rank 0 prints epoch completion
    if rank == 0:
        print(f"finished epoch {epoch+1}")

print("training completed!")
arceus.finish()
