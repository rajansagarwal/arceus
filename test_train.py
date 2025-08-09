import torch
import arceus
from train import Net, DummyDataset
from torch.utils.data import DataLoader


def test_training_loop():
    rank, world_size, args = arceus.cli()
    args.epochs = 1  # Run only one epoch for testing
    
    # setup model
    model = Net()
    model = arceus.wrap(model, show_graph=(rank == 0))
    
    # setup criterion and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    dataloader = DataLoader(DummyDataset(), batch_size=128, shuffle=True)
    
    # training loop
    for epoch in range(args.epochs):
        progress_bar = arceus.progress(dataloader, optimizer)
        
        for data, target in progress_bar:
            data, target = arceus.to_device(data), arceus.to_device(target)
            
            optimizer.zero_grad()
            
            if arceus._USE_AMP:
                device = arceus.get_device()
                with torch.autocast(device_type=device.type, dtype=torch.float16):
                    output = model(data)
                    loss = criterion(output, target)
            else:
                output = model(data)
                loss = criterion(output, target)

            loss.backward()
            optimizer.step()
            
            progress_bar.step(loss=loss)
        
        if rank == 0:
            print(f"finished epoch {epoch+1}")

    print("Test training completed!")
arceus.finish()


if __name__ == "__main__":
    test_training_loop()
