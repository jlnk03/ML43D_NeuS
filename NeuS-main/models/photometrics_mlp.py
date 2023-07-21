import torch
import torch.nn as nn
import torch.nn.functional as F

class PhotometricsMLP(nn.Module):
    def __init__(self, input_dim):
        super(PhotometricsMLP, self).__init__()
        hidden_dim = 64
        self.fc1 = nn.Linear(input_dim * 3, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, rays_list):
        # Sum them 
        # rays_combined = sum(rays_list) / len(rays_list)  # Weighted average of the rays
        # OR concatanate
        rays_combined = torch.cat(rays_list, dim=1) 

        x = F.relu(self.fc1(rays_combined))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        output = self.output_layer(x)

        return output