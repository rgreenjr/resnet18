import torch
import torch.nn as nn


class Resnet18(nn.Module):
    def __init__(self, image_channels, num_classes):
        super(Resnet18, self).__init__()

        self.relu = nn.ReLU()

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

        # ------------------------------------------------------------
        # 1st residual block: 64 -> 64
        # ------------------------------------------------------------
        self.block1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
        )

        # ------------------------------------------------------------
        # 2nd residual block: 64 -> 128
        # ------------------------------------------------------------
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
        )

        # ------------------------------------------------------------
        # 3rd residual block: 128 -> 256
        # ------------------------------------------------------------
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
        )

        # ------------------------------------------------------------
        # 4th residual block: 256 -> 512
        # ------------------------------------------------------------
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
        )

        # classification layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        batch_size = x.shape[0]
        assert x.shape == (batch_size, 3, 224, 224)

        x = self.layer1(x)
        assert x.shape == (batch_size, 64, 56, 56)

        # ------------------------------------------------------------
        # 1st residual block
        # ------------------------------------------------------------
        identity = x
        x = self.block1(x)
        x += identity
        x = self.relu(x)
        assert x.shape == (batch_size, 64, 56, 56)

        # ------------------------------------------------------------
        # 2nd residual block
        # ------------------------------------------------------------
        identity = x
        x = self.block2(x)
        downsample = nn.Conv2d(
            64,
            128,
            kernel_size=1,
            stride=2,
            padding=0,
            bias=False,
        )
        bn = nn.BatchNorm2d(128)
        identity = bn(downsample(identity))
        x += identity
        x = self.relu(x)
        assert x.shape == (batch_size, 128, 28, 28)

        # ------------------------------------------------------------
        # 3rd residual block
        # ------------------------------------------------------------
        identity = x
        x = self.block3(x)
        downsample = nn.Conv2d(
            128,
            256,
            kernel_size=1,
            stride=2,
            padding=0,
            bias=False,
        )
        bn = nn.BatchNorm2d(256)
        identity = bn(downsample(identity))
        x += identity
        x = self.relu(x)
        assert x.shape == (batch_size, 256, 14, 14)

        # ------------------------------------------------------------
        # 4th residual block
        # ------------------------------------------------------------
        identity = x
        x = self.block4(x)
        downsample = nn.Conv2d(
            256,
            512,
            kernel_size=1,
            stride=2,
            padding=0,
            bias=False,
        )
        bn = nn.BatchNorm2d(512)
        identity = bn(downsample(identity))
        x += identity
        x = self.relu(x)
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
