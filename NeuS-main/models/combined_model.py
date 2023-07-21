"""
Point Transformer V2 Mode (recommend)

Disable Grouped Linear

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

from copy import deepcopy
import math
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from model import PointTransformerSeg

# from fields import NeRF


class CombinedModel(nn.Module):
    def __init__(self, nerf):
        super(CombinedModel, self).__init__()
        self.point_transformer = PointTransformerSeg()
        self.nerf = nerf

    def forward(self, input):
        transformed_input = self.point_transformer(input)
        output = self.nerf(transformed_input)
        return output
