import pytorch_lightning as pl
import torch


class DeblurrerLightningModule(pl.LightningModule):
    def __init__(self, model, learning_rate=1e-3):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate

    def training_step(self, batch):
        out = self.model(batch)
        # TODO: calculate, log and return loss

        loss = None
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
