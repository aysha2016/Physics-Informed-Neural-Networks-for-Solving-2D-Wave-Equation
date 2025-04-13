
import torch.nn as nn
import torch

class PINN2DWave(nn.Module):
    def __init__(self):
        super(PINN2DWave, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, x, t):
        input = torch.cat([x, t], dim=1)
        return self.net(input)
