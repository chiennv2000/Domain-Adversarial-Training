import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(nn.Linear(input_dim, 100),
                                   nn.BatchNorm1d(100),
                                   nn.ReLU(),
                                   nn.Linear(100, 2))
    
    def forward(self, x):
        x = self.model(x)
        return F.log_softmax(x)
