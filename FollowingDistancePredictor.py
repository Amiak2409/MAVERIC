import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class FollowingDistancePredictor(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super(FollowingDistancePredictor, self).__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(input_dim + embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.parameters(), lr=0.01, momentum=0.9)

    def forward(self, x, embed):
        x = torch.cat((x, embed), dim=1)
        return self.fc(x)

    def train(self, inputs, embeds, labels):
        self.optimizer.zero_grad()

        outputs = self.forward(inputs, embeds)
        loss = self.criterion(outputs, labels)

        loss.backward()
        self.optimizer.step()
        return loss