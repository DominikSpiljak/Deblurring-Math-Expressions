import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.res_block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding="same",
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding="same",
            ),
        )

    def forward(self, x):
        return self.res_block(x) + x


class ResidualBlocks(nn.Module):
    def __init__(self, in_channels, num_res_blocks):
        super().__init__()
        self.res_blocks = nn.Sequential(
            *[
                ResidualBlock(in_channels=in_channels, out_channels=in_channels)
                for _ in range(num_res_blocks)
            ]
        )

    def forward(self, x):
        return self.res_blocks(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_res_blocks):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding="same",
        )
        self.res_blocks = ResidualBlocks(out_channels, num_res_blocks)

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x)
        x = self.res_blocks(x)
        return x


class FeatureAttentionModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            padding="same",
        )

    def forward(self, eb_out, scm_out):
        x = eb_out * scm_out
        x = self.conv(x)

        return x + eb_out


class EncoderBlockFeatureAttention(nn.Module):
    def __init__(self, in_channels, out_channels, num_res_blocks):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.feature_attention = FeatureAttentionModule(in_channels=out_channels)
        self.res_blocks = ResidualBlocks(
            in_channels=out_channels, num_res_blocks=num_res_blocks
        )

    def forward(self, eb_out, scm_out):
        conv_out = self.conv(eb_out)
        conv_out = F.relu(conv_out)
        fam_out = self.feature_attention(conv_out, scm_out)
        res_out = self.res_blocks(fam_out)
        return res_out


class AsymetricFeatureFusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            padding="same",
        )
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding="same",
        )

    def forward(self, eb1_out, eb2_out, eb3_out):
        cat_out = torch.cat((eb1_out, eb2_out, eb3_out), dim=1)
        conv1_out = self.conv1(cat_out)
        conv1_out = F.relu(conv1_out)
        conv2_out = self.conv2(conv1_out)
        return conv2_out


class ShallowConvolutionalModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels // 8,
            kernel_size=3,
            padding="same",
        )
        self.conv2 = nn.Conv2d(
            in_channels=out_channels // 8,
            out_channels=out_channels // 4,
            kernel_size=1,
            padding="same",
        )
        self.conv3 = nn.Conv2d(
            in_channels=out_channels // 4,
            out_channels=out_channels // 2,
            kernel_size=3,
            padding="same",
        )
        self.conv4 = nn.Conv2d(
            in_channels=out_channels // 2,
            out_channels=out_channels - in_channels,
            kernel_size=1,
            padding="same",
        )
        self.conv5 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=1,
            padding="same",
        )

    def forward(self, x):
        conv1_out = self.conv1(x)
        conv1_out = F.relu(conv1_out)
        conv2_out = self.conv2(conv1_out)
        conv2_out = F.relu(conv2_out)
        conv3_out = self.conv3(conv2_out)
        conv3_out = F.relu(conv3_out)
        conv4_out = self.conv4(conv3_out)
        conv4_out = F.relu(conv4_out)
        conv5_in = torch.cat((conv4_out, x), dim=1)
        conv5_out = self.conv5(conv5_in)
        conv5_out = F.relu(conv5_out)
        return conv5_out


class DecoderBlockResidual(nn.Module):
    def __init__(self, in_channels, out_channels, num_res_blocks):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding="same",
        )
        self.residual_blocks = ResidualBlocks(
            in_channels=out_channels, num_res_blocks=num_res_blocks
        )

    def forward(self, x):
        conv1_out = self.conv1(x)
        conv1_out = F.relu(conv1_out)
        res_out = self.residual_blocks(conv1_out)
        return res_out


class DecoderBlockConvResidualUpscale(nn.Module):
    def __init__(self, in_channels, out_channels, num_res_blocks):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding="same",
        )
        self.residual_blocks = ResidualBlocks(
            in_channels=out_channels, num_res_blocks=num_res_blocks
        )
        self.conv2 = nn.ConvTranspose2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )

    def forward(self, x):
        conv1_out = self.conv1(x)
        conv1_out = F.relu(conv1_out)
        res_out = self.residual_blocks(conv1_out)
        conv2_out = self.conv2(res_out)
        conv2_out = F.relu(conv2_out)
        return conv2_out, res_out


class DecoderBlockResidualUpscale(nn.Module):
    def __init__(self, in_channels, num_res_blocks):
        super().__init__()
        self.residual_blocks = ResidualBlocks(
            in_channels=in_channels, num_res_blocks=num_res_blocks
        )
        self.conv1 = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )

    def forward(self, x):
        res_out = self.residual_blocks(x)
        conv1_out = self.conv1(res_out)
        conv1_out = F.relu(conv1_out)
        return conv1_out, res_out


class MIMOUnet(nn.Module):
    def __init__(self, num_res_blocks=8):
        super().__init__()

        self.scm2 = ShallowConvolutionalModule(in_channels=3, out_channels=64)
        self.scm3 = ShallowConvolutionalModule(in_channels=3, out_channels=128)

        self.eb1 = EncoderBlock(
            in_channels=3, out_channels=32, num_res_blocks=num_res_blocks
        )
        self.eb2 = EncoderBlockFeatureAttention(
            in_channels=32, out_channels=64, num_res_blocks=num_res_blocks
        )
        self.eb3 = EncoderBlockFeatureAttention(
            in_channels=64, out_channels=128, num_res_blocks=num_res_blocks
        )

        self.aff1 = AsymetricFeatureFusion(in_channels=128 + 64 + 32, out_channels=32)
        self.aff2 = AsymetricFeatureFusion(in_channels=128 + 64 + 32, out_channels=64)

        self.db1 = DecoderBlockResidual(
            in_channels=32 + 64, out_channels=32, num_res_blocks=num_res_blocks
        )
        self.db2 = DecoderBlockConvResidualUpscale(
            in_channels=64 + 128, out_channels=64, num_res_blocks=num_res_blocks
        )
        self.db3 = DecoderBlockResidualUpscale(
            in_channels=128, num_res_blocks=num_res_blocks
        )

        self.out_conv_scale1 = nn.Conv2d(
            in_channels=32, out_channels=3, kernel_size=3, padding="same"
        )
        self.out_conv_scale2 = nn.Conv2d(
            in_channels=64, out_channels=3, kernel_size=3, padding="same"
        )
        self.out_conv_scale3 = nn.Conv2d(
            in_channels=128, out_channels=3, kernel_size=3, padding="same"
        )

    def forward(self, x):
        x_2 = F.interpolate(x, scale_factor=0.5)
        x_4 = F.interpolate(x, scale_factor=0.25)

        eb1_out = self.eb1(x)
        scm2_out = self.scm2(x_2)
        scm3_out = self.scm3(x_4)

        eb2_out = self.eb2(eb1_out, scm2_out)
        eb3_out = self.eb3(eb2_out, scm3_out)

        aff1_out = self.aff1(
            eb1_out,
            F.interpolate(eb2_out, scale_factor=2),
            F.interpolate(eb3_out, scale_factor=4),
        )
        aff2_out = self.aff2(
            F.interpolate(eb1_out, scale_factor=0.5),
            eb2_out,
            F.interpolate(eb3_out, scale_factor=2),
        )

        db3_out, db3_res_out = self.db3(eb3_out)
        db2_out, db2_res_out = self.db2(torch.cat((aff2_out, db3_out), dim=1))
        db1_out = self.db1(torch.cat((aff1_out, db2_out), dim=1))

        out_scale3 = self.out_conv_scale3(db3_res_out)
        out_scale2 = self.out_conv_scale2(db2_res_out)
        out_scale1 = self.out_conv_scale1(db1_out)

        return x + out_scale1, x_2 + out_scale2, x_4 + out_scale3


if __name__ == "__main__":
    model = MIMOUnet()
    x = torch.randn(2, 3, 512, 512)
    out, _ = model(x)
    print(out[0].shape, out[1].shape, out[2].shape)
