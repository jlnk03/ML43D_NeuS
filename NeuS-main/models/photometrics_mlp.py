import torch
import torch.nn as nn
import torch.nn.functional as F

class PhotometricsMLP(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=256):
        super(PhotometricsMLP, self).__init__()
        hidden_dim = 64
        self.fc1 = nn.Linear(input_dim * 3, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, input_dim)

        # Learnable parameters for contrast_factor and brightness_delta
        # self.contrast_factor = nn.Parameter(torch.tensor(1.0))
        # self.brightness_delta = nn.Parameter(torch.tensor(0.0))

    def forward(self, rays_list):
        # Sum them 
        # rays_combined = sum(rays_list) / len(rays_list)  # Weighted average of the rays
        # OR concatanate
        rays_combined = torch.cat(rays_list, dim=1) 

        x = F.relu(self.fc1(rays_combined))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        output = self.output_layer(x)

        # output_clone = output.clone()
        # Apply contrast_factor and brightness_delta adjustments
        # output_clone[:, 6:9] = output_clone[:, 6:9].mul(self.contrast_factor).add(self.brightness_delta)

        return output