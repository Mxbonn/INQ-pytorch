from torch import nn


class LinearMnistNet(nn.Module):
    def __init__(self):
        super(LinearMnistNet, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.features(x)
        return x
