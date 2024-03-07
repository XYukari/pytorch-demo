import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        out = self.bn1(x)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += x
        out = self.relu(out)

        return out


class FusionNet(nn.Module):
    def __init__(self):
        super(FusionNet, self).__init__()

        self.seperator_net = nn.Sequential(
            ResBlock(1, 64),
            ResBlock(64, 64)
        )

        self.fusion_net = nn.Sequential(
            ResBlock(128, 64),
            ResBlock(64, 64)
        )

        self.final_conv = nn.Conv2d(64, 1, 1)

    def forward(self, ir, vi):
        ir_features = self.seperator_net(ir)
        vi_features = self.seperator_net(vi)

        combined_features = torch.cat([ir_features, vi_features], dim=1)

        fusion_features = self.fusion_net(combined_features)

        final_output = self.final_conv(fusion_features)

        return final_output
