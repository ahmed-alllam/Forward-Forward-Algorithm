import torch

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model import Model

def load_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('data', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    return train_loader, test_loader


def create_negative_data(data_loader):
    for x, y in data_loader:
        y = torch.multinomial((torch.ones(10).scatter_(0, y, 0) / 9.0), 1).squeeze()
        yield x, y

def train():
    train_loader, test_loader = load_data()
    model = Model(784, 128, 1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
        for x, y in train_loader:
            loss = model.train(x.view(-1, 784), True, optimizer)
        print('Epoch: {}, Loss: {}'.format(epoch, loss.item()))

        for x, y in create_negative_data(train_loader):
            loss = model.train(x.view(-1, 784), False, optimizer)
        print('Epoch: {}, Loss: {}'.format(epoch, loss.item()))

    # torch.save(model.state_dict(), 'model.pth')