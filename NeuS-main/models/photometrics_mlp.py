import torch
import torch.nn as nn
import torch.nn.functional as F

class PhotometricsMLP(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=128):
        super(PhotometricsMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim * 9, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, colors):
        x = F.relu(self.fc1(colors))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        output = self.output_layer(x)

        return output