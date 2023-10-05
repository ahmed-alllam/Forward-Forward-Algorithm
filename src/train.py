import torch

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import tqdm

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


def create_negative_data():
    # make odd indexes negative (randomly)
    return torch.randint(0, 10, (1,))

def encode_label_in_image(x, y):
    # for the first 10 pixels, encode the label as a one-hot vector
    y = torch.nn.functional.one_hot(y, 10).float()
    x[:, :10] = y

    return x

def train(model, train_loader, optimizer, num_epochs):
    for epoch in tqdm.tqdm(range(num_epochs)):
        for x, y in train_loader:
            x = encode_label_in_image(x.view(-1, 784), y)
            loss = model.train(x, True, optimizer)

            x = encode_label_in_image(x.view(-1, 784), create_negative_data())
            loss += model.train(x, False, optimizer)
        
        print("Epoch {} Finished, Train Accuracy: {}".format(epoch, test(model, train_loader)))

def predict(model, x):
    goodness_for_label = []

    for i in range(10):
        x = encode_label_in_image(x.view(-1, 784), torch.tensor([i]))
        goodness, _ = model(x)
        goodness_for_label += [goodness]
    
    return torch.stack(goodness_for_label, 1).argmax(1)

def test(model, test_loader):
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in test_loader:
            pred = predict(model, x)
            correct += (pred == y).sum().item()
            total += y.size(0)

    return correct / total

def main():
    train_loader, test_loader = load_data()
    model = Model(784, 500, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    train(model, train_loader, optimizer, 1000)
    print("Test Accuracy: {}".format(test(model, test_loader)))

if __name__ == '__main__':
    main()