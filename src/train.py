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

def train(model, train_loader, optimizer):
    for epoch in tqdm.tqdm(range(1)):
        for x, y in train_loader:
            x = encode_label_in_image(x.view(-1, 784), y)
            
            loss = model.train(x, True, optimizer)

            x = encode_label_in_image(x.view(-1, 784), create_negative_data())
            loss += model.train(x, False, optimizer)
        
        print('Epoch: {}, Loss: {}'.format(epoch, loss.item()))

    # torch.save(model.state_dict(), 'model.pth')

def predict(model, x):
    goodness_for_label = torch.zeros(10)

    for i in range(10):
        goodness = 0
        x = encode_label_in_image(x.view(-1, 784), torch.tensor([i]))

        out = model.layer1(x)
        goodness += torch.sum(out ** 2).mean()

        out = model.layer2(out)
        goodness += torch.sum(out ** 2).mean()

        goodness_for_label[i] = goodness
    
    return torch.argmax(goodness_for_label)

def test(model, test_loader):
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in test_loader:
            pred = predict(model, x)
            correct += (pred == y).sum().item()
            total += y.size(0)

    print('Accuracy: {}'.format(correct / total))

def main():
    train_loader, test_loader = load_data()
    model = Model(784, 128, 0.5)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.00001, momentum=0.9)
    train(model, train_loader, optimizer)
    test(model, test_loader)

if __name__ == '__main__':
    main()