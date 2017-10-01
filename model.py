import torch
import torch.nn as nn
from torch.nn import init


class paac_ff(nn.Module):
    
    def __init__(self, min_act):
        super(paac_ff, self).__init__()
        self.min_act = min_act
        
        self.main = nn. Sequential(
                        nn.Conv2d(4, 32, 8, 4),
                        nn.ReLU(True),
                        nn.Conv2d(32, 64, 4, 2),
                        nn.ReLU(True),
                        nn.Conv2d(64, 64, 3, 1),
                        nn.ReLU(True),
                        nn.Conv2d(64, 512, 7, 1),
                        nn.ReLU(True),
                    )

        self.v = nn.Conv2d(512, 1, 1)
        self.policy = nn.Conv2d(512, min_act, 1)
        self.Softmax = nn.Softmax()

        for p in self.main.modules():
            if isinstance(p, nn.Conv2d):
                init.kaiming_uniform(p.weight)
                p.bias.data.zero_()

        init.kaiming_uniform(self.v.weight, a=1.0)
        init.kaiming_uniform(self.policy.weight, a=1.0)
        self.v.bias.data.zero_()
        self.policy.bias.data.zero_()

    def forward(self, x):

        y = self.main(x)

        value = self.v(y).squeeze()
        pi = self.Softmax(self.policy(y).view(-1, self.min_act))

        return value, pi
