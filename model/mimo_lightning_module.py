import pytorch_lightning as pl
import torch.nn.functional as F
import torch
from torch.utils import data


def calculate_l1_loss(generated, ground_truth):
    return F.l1_loss(generated, ground_truth)


def calculate_frequency_reconstruction_loss(generated, ground_truth):
    return F.l1_loss(torch.fft.rfftn(generated), torch.fft.rfftn(ground_truth))


def collate_fn(batch):
    blurred = torch.stack([img["blurred"] for img in batch])
    non_blurred = torch.stack([img["non_blurred"] for img in batch])

    return {"blurred": blurred, "non_blurred": non_blurred}


class MIMOUnetModule(pl.LightningModule):
    def __init__(
        self,
        mimo_unet,
        dataset_train,
        dataset_val,
        num_workers,
        learning_rate=1e-3,
        batch_size=8,
        alpha=0.1,
    ):
        super().__init__()
        self.mimo_unet = mimo_unet
        self.dataset_train = dataset_train
        self.dataset_val = dataset_val
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.alpha = alpha

    def training_step(self, batch, batch_idx):
        out = self.mimo_unet(batch["blurred"])
        gt = [
            batch["non_blurred"],
            F.interpolate(batch["non_blurred"], scale_factor=0.5),
            F.interpolate(batch["non_blurred"], scale_factor=0.25),
        ]
        content_loss = torch.mean(
            [
                calculate_l1_loss(out_scale, gt_scale)
                for out_scale, gt_scale in zip(out, gt)
            ]
        )

        msfr_loss = torch.mean(
            [
                calculate_l1_loss(out_scale, gt_scale)
                for out_scale, gt_scale in zip(out, gt)
            ]
        )

        loss = content_loss + self.alpha * msfr_loss

        self.log("Loss", loss, prog_bar=True)

        return {"blurred": batch["blurred"], "deblurred": out[0].detach(), "loss": loss}

    def validation_step_end(self, outputs):
        self.log_metrics(self.validation_loggers, outputs)

    def on_validation_epoch_end(self):
        self.compute_loggers(self.validation_loggers, self.current_epoch, self.logger)

    def training_step_end(self, outputs):
        self.log_metrics(self.train_loggers, outputs)
        return torch.mean(outputs["loss"])

    def on_train_epoch_end(self):
        self.compute_loggers(self.train_loggers, self.current_epoch, self.logger)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.mimo_unet.parameters(), lr=self.learning_rate)
        return optimizer

    def log_metrics(self, loggers, outputs):
        for logger in loggers:
            logger(outputs)

    def compute_loggers(self, loggers, epoch, logger):
        for logger_ in loggers:
            logger_.compute(epoch, logger)

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

    def val_dataloader(self):
        dataloader = data.DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )

        return dataloader
