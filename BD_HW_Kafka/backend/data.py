import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, random_split
from torchvision import datasets, transforms
from pathlib import Path
import glob

def inference_transforms(image):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    return transform(image).unsqueeze(0)

def augmentation_transform():
    augmentation_transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.RandomResizedCrop(28, scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    return augmentation_transform


def load_mnist_data():
    original_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=augmentation_transform())

    augmented_datasets = []
    num_augmented_samples = 100000 - len(original_dataset) 

    while len(augmented_datasets) * len(original_dataset) < num_augmented_samples:
        augmented_datasets.append(original_dataset)

    full_dataset = ConcatDataset([original_dataset] + augmented_datasets)
    print(f"Total samples in the augmented dataset: {len(full_dataset)}")

    return full_dataset


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64*7*7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
def train_model(model, train_loader, criterion, optimizer, epochs=1):
    model.train()
    for epoch in range(epochs):
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')


def run_train(train_loader):
    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_model(model, train_loader, criterion, optimizer, epochs=1)

    return model


def get_mnist_data(need_train):
    dataset = load_mnist_data()

    batch_size = 64

    assert len(dataset) % batch_size == 0, "Dataset size is not divisible by batch_size"

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"train_loader : {len(train_loader)}")
    print(f"test_loader : {len(test_loader)}")

    save_folder = Path("./outs_and_logs/")
    save_folder.mkdir(parents=False, exist_ok=True)
    curr_num_of_weight = len([file for file in save_folder.iterdir() if file.suffix in ['.pth', 'pt']])
    
    if need_train:
        model = run_train(train_loader)
        torch.save(model.state_dict(), path_to_save_weights)
        save_name = "mnist_cnn.pth" if curr_num_of_weight == 0 else f"mnist_cnn_{curr_num_of_weight}.pth"
    else:
        save_name = "mnist_cnn.pth"

    path_to_save_weights = save_folder / save_name
    return test_loader, path_to_save_weights

# if __name__ == "__main__":
#     test_loader, path_to_save_weights = get_mnist_data()
#     print(f"Done! Save path: {path_to_save_weights}")