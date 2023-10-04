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

def encode_label_in_image(x, y):
    # for the first 10 pixels, encode the label as a one-hot vector
    y = torch.nn.functional.one_hot(y, 10).float().view(-1, 10, 1)
    x[:, :10, :] = y
    return x

def train(model, train_loader, optimizer):
    for epoch in range(10):
        for x, y in train_loader:
            encode_label_in_image(x, y)
            loss = model.train(x, True, optimizer)
        
        print('Epoch: {}, Loss: {}'.format(epoch, loss.item()))

        for x, y in create_negative_data(train_loader):
            x = encode_label_in_image(x, y)
            loss = model.train(x, False, optimizer)
        
        print('Epoch: {}, Loss: {}'.format(epoch, loss.item()))

    # torch.save(model.state_dict(), 'model.pth')

def test(model, test_loader):
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in test_loader:
            # do one against all by 10 forward passes
            for i in range(10):
                out = model(encode_label_in_image(x, i))
                out = torch.argmax(out, dim=1)
            
            correct += torch.sum(out == y).item()
            total += len(y)

    print('Accuracy: {}'.format(correct / total))

def main():
    train_loader, test_loader = load_data()
    model = Model(784, 128, 0.5)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    train(model, train_loader, optimizer)
    test(model, test_loader)

if __name__ == '__main__':
    main()