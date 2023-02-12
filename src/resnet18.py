import torch
import torch.nn as nn

from src.block import Block


class Resnet18(nn.Module):
    def __init__(self, image_channels, num_classes):
        super(Resnet18, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(
                image_channels,
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # residual blocks
        self.block1 = Block(64, 64, stride=1)
        self.block2 = Block(64, 128, stride=2)
        self.block3 = Block(128, 256, stride=2)
        self.block4 = Block(256, 512, stride=2)

        # classification layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        batch_size = x.shape[0]
        assert x.shape == (batch_size, 3, 224, 224)

        x = self.layer1(x)
        assert x.shape == (batch_size, 64, 56, 56)

        x = self.block1(x)
        assert x.shape == (batch_size, 64, 56, 56)

        x = self.block2(x)
        assert x.shape == (batch_size, 128, 28, 28)

        x = self.block3(x)
        assert x.shape == (batch_size, 256, 14, 14)

        x = self.block4(x)
        assert x.shape == (batch_size, 512, 7, 7)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


if __name__ == "__main__":
    model = Resnet18(3, 10)
    x = torch.ones(1, 3, 224, 224)
    print(model)
    print(model(x).shape)
