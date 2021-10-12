from torchvision.models import vgg19
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = vgg19(num_classes=1)

    def forward(self, batch):
        return self.model.forward(batch)
