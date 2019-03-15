from torch import nn


class MnistNet(nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0),
            nn.ReLU(True),
        )

        self.width_last_channel = 1152

        self.classifier = nn.Linear(1152, 10)

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.width_last_channel)
        x = self.classifier(x)
        return x
