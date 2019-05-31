import torch
import torch.nn as nn

def init_linear_weights_xavier(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)

class Net(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(input_size[0], 32, 8, 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(),
        )

        flatten_size = self.cnn(torch.zeros(1, *input_size)).view(-1).size(0)

        self.fcs = nn.Sequential(
            nn.Linear(flatten_size, 512),
            nn.ReLU(),
        )
        self.policy = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
            nn.Softmax(),
        )
        self.value = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        ) 

        self.apply(init_linear_weights_xavier)

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.fcs(x)

        policy = self.policy(x)
        value = self.value(x)

        return policy, value