import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureExtractor(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size):
        super(FeatureExtractor, self).__init__()
        self.model = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size),
                                   nn.BatchNorm2d(64),
                                   nn.MaxPool2d(2),
                                   nn.ReLU(),
                                   nn.Conv2d(64, out_channels, kernel_size),
                                   nn.BatchNorm2d(out_channels),
                                   nn.MaxPool2d(2),
                                   nn.ReLU())
    
    def forward(self, x):
        x = self.model(x)
        return x
    
