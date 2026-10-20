import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("./runs/cnn_experiment")


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


def histogram(model: SimpleCNN, step) -> None:

    for name, param in model.named_parameters():
        if param.grad is not None:
            writer.add_histogram(f"gradients/{name}", param.grad, step)


def evaluate(
    dataloader: DataLoader,
    model: SimpleCNN,
    opt: Adam,
    device,
    epoch,
    train: bool = True,
) -> tuple[float, float]:
    run_loss = 0.0
    correct = 0
    total = 0
    if train:
        criterion = nn.CrossEntropyLoss()

        model.train()

        for i, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            opt.zero_grad()
            outputs = model(inputs)
            loss = criterion.forward(outputs, targets)
            loss.backward()
            # histogram(model, epoch * len(trainloader) + i)
            opt.step()
            run_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

    else:
        model.eval()
        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            for i, (inps, trgs) in enumerate(dataloader):
                inputs, targets = inps.to(device), trgs.to(device)
                outputs = model(inputs)
                loss = criterion.forward(outputs, targets)
                run_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)

    return (run_loss / len(dataloader)), ((correct / total) * 100)


# Expreiment with
# - X zero-init weights
# - X no BatchNorm
# - X learning rate 10.0
# - X no normalization
# - X slow dataloader
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
    testloader = DataLoader(testset, batch_size=64, shuffle=True, num_workers=8)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = SimpleCNN().to(device)
    opt = Adam(model.parameters(), lr=0.01)

    epochs = 5
    for e in range(epochs):
        running_loss_train, percent_train = evaluate(trainloader, model, opt, device, e)
        running_loss_test, percent_test = evaluate(
            testloader, model, opt, device, e, train=False
        )

        print(f"{e}: train={int(percent_train)}% -> test={int(percent_test)}%")
