import torch
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

        self.adapool = nn.AdaptiveAvgPool2d((7, 7))

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 1),
        )

    def forward(self, batch):
        x = self.conv1(batch)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pooling1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.pooling2(x)

        x = self.conv5(x)
        x = F.relu(x)
        x = self.conv6(x)
        x = F.relu(x)
        x = self.conv7(x)
        x = F.relu(x)
        x = self.conv8(x)
        x = F.relu(x)
        x = self.pooling3(x)

        x = self.conv9(x)
        x = F.relu(x)
        x = self.conv10(x)
        x = F.relu(x)
        x = self.conv11(x)
        x = F.relu(x)
        x = self.conv12(x)
        x = F.relu(x)
        x = self.pooling4(x)

        x = self.conv13(x)
        x = F.relu(x)
        x = self.conv14(x)
        x = F.relu(x)
        x = self.conv15(x)
        x = F.relu(x)
        x = self.conv16(x)
        x = F.relu(x)
        x = self.pooling5(x)

        x = self.adapool(x)
        x = torch.flatten(x, 1)

        x = self.classifier(x)

        return x
