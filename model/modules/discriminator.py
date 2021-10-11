from torchvision.models import vgg19


class Discriminator(nn.Module):
    def __init__(self):
        self.model = vgg19(num_classes=2)

    def forward(self, batch):
        return self.model.forward(batch)
