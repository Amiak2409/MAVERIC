import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class MutualInformation(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super(MutualInformation, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim)
        )
        
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.parameters(), lr=0.01, momentum=0.9)

    def forward(self, X, P):
        x = torch.cat((X, P), dim=1)
        return self.fc(x)
    
    def train(self, X, P, labels):
        self.optimizer.zero_grad()

        outputs = self.forward(X, P)
        loss = self.criterion(outputs, labels)

        loss.backward()
        self.optimizer.step()

        print(f"Loss: {loss.item()}")


