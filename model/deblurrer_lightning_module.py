import pytorch_lightning as pl
import torch.nn.functional as F
import torch
from torch.utils import data

from collections import OrderedDict


def calculate_loss(predictions, real):
    return F.binary_cross_entropy_with_logits(predictions, real)


class DeblurrerLightningModule(pl.LightningModule):
    def __init__(
        self,
        db_generator,
        discriminator,
        non_blurred_dataset,
        blurred_dataset,
        num_workers,
        learning_rate=1e-3,
        batch_size=8,
    ):
        super().__init__()
        self.db_generator = db_generator
        self.db_discriminator = discriminator
        self.non_blurred_dataset = non_blurred_dataset
        self.blurred_dataset = blurred_dataset
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.loss_f = calculate_loss

    def training_step(self, batch, batch_idx, optimizer_idx):

        if optimizer_idx == 0:
            clear_labels = torch.ones(batch["blurred"].size(0), 1)
            self.deblurred = self.db_generator(batch["blurred"])

            prediction = self.db_discriminator(self.deblurred)
            g_loss = self.loss_f(prediction, clear_labels)

            tqdm_dict = {"g_loss": g_loss.detach()}
            output = OrderedDict(
                {
                    "loss": g_loss,
                    "progress_bar": tqdm_dict,
                    "log": tqdm_dict,
                    "g_loss": g_loss.detach(),
                }
            )
            return output

        if optimizer_idx == 1:
            fake_labels = torch.zeros(batch["non_blurred"].size(0), 1)
            real_labels = torch.ones(batch["non_blurred"].size(0), 1)

            prediction = self.db_discriminator(batch["non_blurred"])
            d_real_loss = self.loss_f(prediction, real_labels)

            prediction = self.db_discriminator(self.deblurred.detach())
            d_fake_loss = self.loss_f(prediction, fake_labels)

            d_loss = (d_real_loss + d_fake_loss) / 2.0

            tqdm_dict = {"d_loss": d_loss.detach()}
            output = OrderedDict(
                {
                    "loss": d_loss,
                    "progress_bar": tqdm_dict,
                    "log": tqdm_dict,
                    "d_loss": d_loss.detach(),
                }
            )
            return output

    def configure_optimizers(self):
        optimizer_dbG = torch.optim.Adam(
            self.db_generator.parameters(), lr=self.learning_rate
        )
        optimizer_dbD = torch.optim.Adam(
            self.db_discriminator.parameters(), lr=self.learning_rate
        )
        return [optimizer_dbG, optimizer_dbD]

    def train_dataloader(self):
        non_blurred_dataloader = data.DataLoader(
            self.non_blurred_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        blurred_dataloader = data.DataLoader(
            self.blurred_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        return {"non_blurred": non_blurred_dataloader, "blurred": blurred_dataloader}
