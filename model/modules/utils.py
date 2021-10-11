import torch.nn as nn
import torch


class BuildingBlock(nn.Module):
    def __init__(self, features=64, kernel_size=3, lrelu_slope=0.01):
        self.building_block = nn.Sequential(
            [
                nn.Conv2d(features, features, kernel_size, padding="same"),
                nn.LeakyReLU(lrelu_slope),
            ]
        )

    def forward(self, batch):
        return self.building_block.forward(batch)


class ResBlock(nn.Module):
    def __init__(self, features=64, kernel_size=3, lrelu_slope=0.01):
        self.block = nn.Sequential(
            [
                *[
                    BuildingBlock(
                        features=features,
                        kernel_size=kernel_size,
                        lrelu_slope=lrelu_slope,
                    )
                    for _ in range(4)
                ],
                nn.Conv2d(features, features, kernel_size, padding="same"),
            ]
        )

    def forward(self, batch):
        return self.block.forward(batch) + batch
