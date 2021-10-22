import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, padding=1
        )
        self.pooling1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, padding=1
        )
        self.conv4 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=3, padding=1
        )
        self.pooling2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=3, padding=1
        )
        self.conv6 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=3, padding=1
        )
        self.conv7 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=3, padding=1
        )
        self.conv8 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=3, padding=1
        )
        self.pooling3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv9 = nn.Conv2d(
            in_channels=256, out_channels=512, kernel_size=3, padding=1
        )
        self.conv10 = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=3, padding=1
        )
        self.conv11 = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=3, padding=1
        )
        self.conv12 = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=3, padding=1
        )
        self.pooling4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv13 = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=3, padding=1
        )
        self.conv14 = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=3, padding=1
        )
        self.conv15 = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=3, padding=1
        )
        self.conv16 = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=3, padding=1
        )
        self.pooling5 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, batch):
        x = self.conv1(batch)
        x = F.relu(x)
        x = self.conv2(batch)
        x = F.relu(x)
        x = self.pooling1(x)

        x = self.conv3(batch)
        x = F.relu(x)
        x = self.conv4(batch)
        x = F.relu(x)
        x = self.pooling2(x)

        x = self.conv5(batch)
        x = F.relu(x)
        x = self.conv6(batch)
        x = F.relu(x)
        x = self.conv7(batch)
        x = F.relu(x)
        x = self.conv8(batch)
        x = F.relu(x)
        x = self.pooling3(batch)

        x = self.conv9(batch)
        x = F.relu(x)
        x = self.conv10(batch)
        x = F.relu(x)
        x = self.conv11(batch)
        x = F.relu(x)
        x = self.conv12(batch)
        x = F.relu(x)
        x = self.pooling4(batch)

        x = self.conv13(batch)
        x = F.relu(x)
        x = self.conv14(batch)
        x = F.relu(x)
        x = self.conv15(batch)
        x = F.relu(x)
        x = self.conv16(batch)
        x = F.relu(x)
        x = self.pooling5(batch)

        return x
