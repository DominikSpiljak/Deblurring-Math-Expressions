import torch.nn as nn
from model.modules.utils import ResBlock
import torch


class DBGenerator(nn.Module):
    def __init__(
        self,
        input_channels=3,
        num_res_blocks=16,
        res_blocks_dim=63,
        res_kernel_size=3,
        lrelu_slope=0.01,
    ):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(input_channels, res_blocks_dim, res_kernel_size, padding="same"),
            *[
                ResBlock(
                    features=res_blocks_dim,
                    kernel_size=res_kernel_size,
                    lrelu_slope=lrelu_slope,
                )
                for _ in range(num_res_blocks)
            ],
            nn.Conv2d(res_blocks_dim, res_blocks_dim, res_kernel_size, padding="same"),
            nn.LeakyReLU(lrelu_slope),
            nn.Conv2d(res_blocks_dim, 3, res_kernel_size, padding="same"),
            nn.Tanh()
        )

    def forward(self, batch):
        return self.model(batch)
