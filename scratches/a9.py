import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision

class SimpleCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))

def evaluate(dataloader:DataLoader, model: SimpleCNN,opt:Adam, device,  train: bool = True) -> float:
    run_loss = 0.0
    if train:
        criterion = nn.CrossEntropyLoss()

        model.train()

        for i, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            opt.zero_grad()
            outputs = model(inputs)
            loss = criterion.forward(outputs, targets)
            loss.backward()

            opt.step()
            run_loss += loss.item()

    else:
        model.eval()
        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            for i, (inps, trgs) in enumerate(dataloader):
                inputs, targets = inps.to(device), trgs.to(device)
                outputs = model(inputs)
                loss = criterion.forward(outputs, targets)
                run_loss += loss.item()
        
    return run_loss/len(dataloader)
   
# Expreiment with
# - zero-init weights
# - no BatchNorm
# - learning rate 10.0
# - no normalization
# - slow dataloader
# Record loss curves and gradient histograms, and read the PyTorch Profiler trace.
if __name__ == "__main__":
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
            ),  # CIFAR-10 means & stds
        ]
    )

    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=False, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=64, shuffle=True, num_workers=8
    )

    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=False, transform=transform
    )
    testloader = DataLoader(
        testset, batch_size=64, shuffle=True, num_workers=8
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = SimpleCNN().to(device)
    opt = Adam(model.parameters())

    epochs = 5
    for e in range(epochs):
        running_loss_train= evaluate(trainloader, model,opt , device)
        running_loss_test = evaluate(testloader, model, opt, device,train = False)
        
        print(f"{e}| {running_loss_train:.2f} -> {running_loss_test:.2f}")

