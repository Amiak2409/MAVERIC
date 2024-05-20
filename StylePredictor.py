import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class StylePredictor(nn.Module):
    def __init__(self, embed_dim):
        super(StylePredictor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, 1)
        )
        
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.parameters(), lr=0.01, momentum=0.9)

    def forward(self, embed):
        return self.fc(embed)
    
    def train(self, labels, embeds):
        self.optimizer.zero_grad()

        outputs = self.forward(embeds)
        loss = self.criterion(outputs, labels)

        loss.backward()
        self.optimizer.step()

        print(f"Loss: {loss.item()}")



