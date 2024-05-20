import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class VelocityPredictor(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super(VelocityPredictor, self).__init__()
        self.lstm = nn.LSTM(input_dim, 128)
        self.fc = nn.Sequential(
            nn.Linear(input_dim + embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.parameters(), lr=0.01, momentum=0.9)

    def forward(self, x, embed):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[-1, :, :]
        x = torch.cat((lstm_out, embed), dim=1)
        return lstm_out, self.fc(x)
    
    def train(self, inputs, labels, embeds):
        self.optimizer.zero_grad()

        outputs = self.forward(inputs, embeds)
        loss = self.criterion(outputs, labels)

        loss.backward()
        self.optimizer.step()

        print(f"Loss: {loss.item()}")



