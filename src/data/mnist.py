import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from torchvision.datasets import MNIST
import torchvision.transforms as transforms


def get_data_loaders(root_dir='./data', train_batch_size=128, validation_batch_size=1000):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307, ), (0.3081, )),
    ])

    train_loader = DataLoader(MNIST(root=root_dir, train=True, transform=transform, download=True),
                              batch_size=train_batch_size, shuffle=True)
    validation_loader = DataLoader(MNIST(root=root_dir, train=False, transform=transform, download=True),
                                   batch_size=validation_batch_size, shuffle=True)
    return train_loader, validation_loader


def get_train_val_test_loaders(root_dir='/data', train_batch_size=128, validation_batch_size=128, test_batch_size=1000,
                               seed=42):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307, ), (0.3081, )),
    ])

    train_validation_dataset = MNIST(root=root_dir, train=True, transform=transform, download=True)
    test_dataset = MNIST(root=root_dir, train=False, transform=transform, download=True)
    train_indices, validation_indices = train_test_split(range(60000), test_size=10000, shuffle=True,
                                                         random_state=seed)
    train_dataset = torch.utils.data.Subset(train_validation_dataset, train_indices)
    validation_dataset = torch.utils.data.Subset(train_validation_dataset, validation_indices)

    train_loader = DataLoader(train_dataset,
                              batch_size=train_batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset,
                                   batch_size=validation_batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset,
                             batch_size=test_batch_size, shuffle=True)
    return train_loader, validation_loader, test_loader


def detransfrom(x_batch):
    x_batch = x_batch * 0.3081 + 0.1307
    x_batch = x_batch.clamp(0, 1)
    x_batch = x_batch.view(x_batch.size(0), 1, 28, 28)
    return x_batch.numpy()
