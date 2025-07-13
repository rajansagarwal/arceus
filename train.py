import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import arceus

# CLI setup
rank, world_size, args = arceus.cli()

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # simple fully connected network
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
        # random 28x28 image and label
        return torch.randn(1, 28, 28), torch.randint(0, 10, (1,)).item()

# setup model and distributed wrapper
model = Net()
model = arceus.wrap(model, show_graph=(rank == 0))

# device setup
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

# training setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())
dataloader = DataLoader(DummyDataset(), batch_size=128, shuffle=True)

# training loop
for epoch in range(args.epochs):
    progress_bar = arceus.progress(dataloader)
    
    for data, target in progress_bar:
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        
        # AMP if enabled
        if arceus._USE_AMP:
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                loss = criterion(output, target)
        else:
            loss = criterion(output, target)
        
        loss.backward()
        optimizer.step()
    
    # only rank 0 prints epoch completion
    if rank == 0:
        print(f"Finished epoch {epoch+1}")

print("Training completed!")
arceus.finish()
