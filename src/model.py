import torch

class Model(torch.nn.Module):
    def __init__(self, input_size, hidden_size, threshold):
        super(Model, self).__init__()
        self.layer1 = torch.nn.Linear(input_size, hidden_size)
        self.layer2 = torch.nn.Linear(hidden_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.threshold = threshold
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.relu(out)
        out = self.layer2(out)
        return out

    def train(self, x, is_positive, optimizer):        
        out = self.layer1(x)
        loss = torch.sum(out ** 2) - self.threshold if is_positive else self.threshold - torch.sum(out ** 2)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        optimizer.zero_grad()

        out = self.layer2(out)
        loss = torch.sum(out ** 2) - self.threshold if is_positive else self.threshold - torch.sum(out ** 2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss
