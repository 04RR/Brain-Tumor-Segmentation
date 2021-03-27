import torch
import torch.nn as nn


class UNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()

        self.conv1 = self.contract_block(in_channels, 32, 7, 3)
        self.conv2 = self.contract_block(32, 64, 3, 1)
        self.conv3 = self.contract_block(64, 64, 3, 1)
        self.conv4 = self.contract_block(64, 64, 3, 1)
        self.conv5 = self.contract_block(64, 64, 3, 1)
        self.conv6 = self.contract_block(64, 64, 3, 1)
        self.conv7 = self.contract_block(64, 64, 3, 1)
        self.conv8 = self.contract_block(64, 64, 3, 1)

        self.upconv8 = self.expand_block(64, 64, 3, 1)
        self.upconv7 = self.expand_block(128, 64, 3, 1)
        self.upconv6 = self.expand_block(128, 64, 3, 1)
        self.upconv5 = self.expand_block(128, 64, 3, 1)
        self.upconv4 = self.expand_block(128, 64, 3, 1)
        self.upconv3 = self.expand_block(128, 64, 3, 1)
        self.upconv2 = self.expand_block(128, 32, 3, 1)
        self.upconv1 = self.expand_block(64, out_channels, 3, 1)

    def __call__(self, x):

        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        conv6 = self.conv6(conv5)
        conv7 = self.conv7(conv6)
        conv8 = self.conv8(conv7)

        upconv8 = self.upconv8(conv8)

        upconv7 = self.upconv7(torch.cat([upconv8, conv7], 1))
        upconv6 = self.upconv6(torch.cat([upconv7, conv6], 1))
        upconv5 = self.upconv5(torch.cat([upconv6, conv5], 1))
        upconv4 = self.upconv4(torch.cat([upconv5, conv4], 1))
        upconv3 = self.upconv3(torch.cat([upconv4, conv3], 1))
        upconv2 = self.upconv2(torch.cat([upconv3, conv2], 1))
        upconv1 = self.upconv1(torch.cat([upconv2, conv1], 1))

        return upconv1

    def contract_block(self, in_channels, out_channels, kernel_size, padding):

        contract = nn.Sequential(
            torch.nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
            ),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
            ),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
        )

        return contract

    def expand_block(self, in_channels, out_channels, kernel_size, padding):

        expand = nn.Sequential(
            torch.nn.Conv2d(
                in_channels, out_channels, kernel_size, stride=1, padding=padding
            ),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                out_channels, out_channels, kernel_size, stride=1, padding=padding
            ),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
        )
        return expand


def dice_coef_loss(inputs, target):
    intersection = 2.0 * ((target * inputs).sum())
    union = target.sum() + inputs.sum()
    return 1 - (intersection / union)


def mse_dice_loss(inputs, target):
    loss_function = nn.MSELoss()
    dicescore = dice_coef_loss(inputs, target)
    bceloss = loss_function(inputs, target)
    return bceloss + dicescore * 1.25

