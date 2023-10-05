import torch

class Model(torch.nn.Module):
    def __init__(self, input_size, hidden_size, threshold):
        super(Model, self).__init__()
        self.layer1 = torch.nn.Linear(input_size, hidden_size)
        self.layer2 = torch.nn.Linear(hidden_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.threshold = threshold

        self.layers = [self.layer1, self.layer2]
    
    def forward(self, x):
        goodness = []

        out = x
        for layer in self.layers:
            out = layer(out)
            out = self.relu(out)
            goodness += [out.pow(2).mean(1)]

        return torch.stack(goodness, 1).sum(1), out

    def train(self, x, is_positive, optimizer):        
        out = x

        for layer in self.layers:
            out = layer(out.detach())
            out = self.relu(out)

            loss = out.pow(2).mean(1) - self.threshold
            loss = -loss if is_positive else loss
            loss = torch.log(1 + torch.exp(loss)).mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return loss
