import torch
import torch.nn as nn


class Resnet18(nn.Module):
    def __init__(self, imgae_channels, num_classes):
        super(Resnet18, self).__init__()

        # ------------------------------------------------------------
        # conv 7x7, in_channels=3, out_channels=64, stride 2, padding 3
        # ------------------------------------------------------------
        self.conv1 = nn.Conv2d(
            in_channels=imgae_channels,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

        # maxpool 3x3, stride 2, padding 1
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ------------------------------------------------------------
        # 1st residual block: 3x3, in_channels 64, out_channels 64
        # ------------------------------------------------------------
        self.layer1_conv1 = nn.Conv2d(
            64, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.layer1_bn1 = nn.BatchNorm2d(64)
        self.layer1_conv2 = nn.Conv2d(
            64, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.layer1_bn2 = nn.BatchNorm2d(64)

        # ------------------------------------------------------------
        # 2nd residual block: 3x3, in_channels 64, out_channels 128
        # ------------------------------------------------------------
        self.layer2_conv1 = nn.Conv2d(
            64, 128, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.layer2_bn1 = nn.BatchNorm2d(128)
        self.layer2_conv2 = nn.Conv2d(
            128, 128, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.layer2_bn2 = nn.BatchNorm2d(128)

        # ------------------------------------------------------------
        # 3rd residual block: 3x3, in_channels 128, out_channels 256
        # ------------------------------------------------------------
        self.layer3_conv1 = nn.Conv2d(
            128, 256, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.layer3_bn1 = nn.BatchNorm2d(256)
        self.layer3_conv2 = nn.Conv2d(
            256, 256, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.layer3_bn2 = nn.BatchNorm2d(256)

        # ------------------------------------------------------------
        # 4th residual block: 3x3, in_channels 256, out_channels 512
        # ------------------------------------------------------------
        self.layer4_conv1 = nn.Conv2d(
            256, 512, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.layer4_bn1 = nn.BatchNorm2d(512)
        self.layer4_conv2 = nn.Conv2d(
            512, 512, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.layer4_bn2 = nn.BatchNorm2d(512)

        # avgpool 7x7, stride 1
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # fully connected
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        batch_size = x.shape[0]
        assert x.shape == (batch_size, 3, 224, 224)

        x = self.conv1(x)
        assert x.shape == (batch_size, 64, 112, 112)

        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        assert x.shape == (batch_size, 64, 56, 56)

        # ------------------------------------------------------------
        # 1st residual block
        # ------------------------------------------------------------
        identity = x
        x = self.layer1_conv1(x)
        x = self.layer1_bn1(x)
        x = self.relu(x)
        x = self.layer1_conv2(x)
        x = self.layer1_bn2(x)
        x += identity
        x = self.relu(x)
        assert x.shape == (batch_size, 64, 56, 56)

        # ------------------------------------------------------------
        # 2nd residual block
        # ------------------------------------------------------------
        identity = x
        x = self.layer2_conv1(x)
        x = self.layer2_bn1(x)
        x = self.relu(x)
        x = self.layer2_conv2(x)
        x = self.layer2_bn2(x)
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
        x = self.layer3_conv1(x)
        x = self.layer3_bn1(x)
        x = self.relu(x)
        x = self.layer3_conv2(x)
        x = self.layer3_bn2(x)
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
        x = self.layer4_conv1(x)
        x = self.layer4_bn1(x)
        x = self.relu(x)
        x = self.layer4_conv2(x)
        x = self.layer4_bn2(x)
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
