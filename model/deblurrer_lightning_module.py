import pytorch_lightning as pl
import torch.nn.functional as F
import torch
from torch.utils import data


def calculate_bce_loss(predictions, real):
    return F.binary_cross_entropy_with_logits(predictions, real)


def calculate_l1_loss(generated, ground_truth):
    return F.mse_loss(generated, ground_truth)


def collate_fn(batch):
    blurred = torch.stack([img["blurred"] for img in batch])
    non_blurred = torch.stack([img["non_blurred"] for img in batch])

    return {"blurred": blurred, "non_blurred": non_blurred}


class DeblurrerLightningModule(pl.LightningModule):
    def __init__(
        self,
        db_generator,
        discriminator,
        dataset_train,
        dataset_val,
        num_workers,
        learning_rate=1e-3,
        batch_size=8,
        alpha=0.1,
    ):
        super().__init__()
        self.db_generator = db_generator
        self.db_discriminator = discriminator
        self.dataset_train = dataset_train
        self.dataset_val = dataset_val
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.bce_loss = calculate_bce_loss
        self.l1_loss = calculate_l1_loss
        self.alpha = alpha

    def training_step(self, batch, batch_idx, optimizer_idx):

        if optimizer_idx == 0:
            clear_labels = torch.ones(batch["blurred"].size(0), 1, device=self.device)
            deblurred = self.db_generator(batch["blurred"])

            prediction = self.db_discriminator(deblurred)
            g_loss_bce = self.bce_loss(prediction, clear_labels)
            g_loss_l1 = self.l1_loss(deblurred, batch["non_blurred"])

            g_loss = g_loss_l1 + g_loss_bce * self.alpha
            self.log("Generator loss", g_loss, prog_bar=True)

            return {
                "loss": g_loss,
                "blurred": batch["blurred"],
                "deblurred": deblurred.detach(),
                "optimizer_idx": optimizer_idx,
            }

        if optimizer_idx == 1:
            fake_labels = torch.zeros(
                batch["non_blurred"].size(0), 1, device=self.device
            )
            real_labels = torch.ones(
                batch["non_blurred"].size(0), 1, device=self.device
            )

            prediction = self.db_discriminator(batch["non_blurred"])
            d_real_loss = self.bce_loss(prediction, real_labels)

            prediction = self.db_discriminator(
                self.db_generator(batch["blurred"]).detach()
            )
            d_fake_loss = self.bce_loss(prediction, fake_labels)

            d_loss = (d_real_loss + d_fake_loss) / 2.0

            self.log("Discriminator loss", d_loss, prog_bar=True)

            return {"loss": d_loss, "optimizer_idx": optimizer_idx}

    def training_step_end(self, outputs):
        self.log_metrics(self.train_loggers, outputs)

    def on_train_epoch_end(self):
        self.compute_loggers(self.train_loggers, self.current_epoch)

    def configure_optimizers(self):
        optimizer_dbG = torch.optim.Adam(
            self.db_generator.parameters(), lr=self.learning_rate
        )
        optimizer_dbD = torch.optim.Adam(
            self.db_discriminator.parameters(), lr=self.learning_rate
        )
        return [optimizer_dbG, optimizer_dbD]

    def log_metrics(loggers, outputs):
        for logger in loggers:
            logger(outputs)

    def compute_loggers(loggers, epoch):
        for logger in loggers:
            logger.compute(epoch)

    def train_dataloader(self):
        dataloader = data.DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )

        return dataloader
