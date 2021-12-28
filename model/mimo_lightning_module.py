import logging
from argparse import Namespace

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils import data

from data.dataset import get_dataset
from loggers.loggers import ImageLogger
from model.mimo_unet_modules.mimo_unet import MIMOUnet


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
        *,
        data_args,
        model_args,
        training_args,
        logger_args,
    ):
        super().__init__()

        if not isinstance(model_args, Namespace):
            model_args = Namespace(**model_args)

        self.data_args = data_args
        self.model_args = model_args
        self.training_args = training_args
        self.logger_args = logger_args
        self.model = MIMOUnet(
            **{k: v for k, v in vars(self.model_args).items() if v is not None}
        )
        self.dataset_train, self.dataset_val = self.setup_datasets()
        self.setup_loggers()
        self.save_hyperparameters("model_args")

    def setup_loggers(self):
        self.train_loggers = []
        self.validation_loggers = []

        if not self.logger_args.disable_image_logging:
            self.validation_loggers.append(
                ImageLogger(
                    self.logger_args.max_batches_logged_per_epoch,
                    self.training_args.batch_size,
                )
            )

    def setup_datasets(self):
        return get_dataset(
            self.data_args.dataset,
            self.data_args.img_size,
            kernel_size=self.data_args.kernel_size,
            sigmas=self.data_args.sigmas,
        )

    def training_step(self, batch, batch_idx):
        out = self.model(batch["blurred"])
        gt = [
            batch["non_blurred"],
            F.interpolate(batch["non_blurred"], scale_factor=0.5),
            F.interpolate(batch["non_blurred"], scale_factor=0.25),
        ]
        content_loss = (
            calculate_l1_loss(out[0], gt[0])
            + calculate_l1_loss(out[1], gt[1])
            + calculate_l1_loss(out[2], gt[2])
        ) / 3

        msfr_loss = (
            calculate_frequency_reconstruction_loss(out[0], gt[0])
            + calculate_frequency_reconstruction_loss(out[1], gt[1])
            + calculate_frequency_reconstruction_loss(out[2], gt[2])
        ) / 3

        loss = content_loss + self.training_args.alpha * msfr_loss

        return {
            "blurred": batch["blurred"],
            "deblurred": out[0].detach(),
            "non_blurred": batch["non_blurred"],
            "loss": loss,
        }

    def validation_step(self, batch, batch_idx):
        out = self.model(batch["blurred"])
        gt = [
            batch["non_blurred"],
            F.interpolate(batch["non_blurred"], scale_factor=0.5),
            F.interpolate(batch["non_blurred"], scale_factor=0.25),
        ]
        content_loss = (
            calculate_l1_loss(out[0], gt[0])
            + calculate_l1_loss(out[1], gt[1])
            + calculate_l1_loss(out[2], gt[2])
        ) / 3

        msfr_loss = (
            calculate_frequency_reconstruction_loss(out[0], gt[0])
            + calculate_frequency_reconstruction_loss(out[1], gt[1])
            + calculate_frequency_reconstruction_loss(out[2], gt[2])
        ) / 3

        loss = content_loss + self.training_args.alpha * msfr_loss

        return {
            "blurred": batch["blurred"],
            "deblurred": out[0].detach(),
            "non_blurred": batch["non_blurred"],
            "loss": loss,
        }

    def validation_step_end(self, outputs):
        self.log_metrics(self.validation_loggers, outputs)
        loss = torch.mean(outputs["loss"])
        self.log("Validation loss", loss)
        return loss

    def on_validation_epoch_end(self):
        self.compute_loggers(self.validation_loggers, self.current_epoch, self.logger)

    def training_step_end(self, outputs):
        self.log_metrics(self.train_loggers, outputs)
        loss = torch.mean(outputs["loss"])
        self.log("Train loss", loss)
        return loss

    def on_train_epoch_end(self):
        self.compute_loggers(self.train_loggers, self.current_epoch, self.logger)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.training_args.learning_rate
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.9, patience=3, threshold=1e-5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "Validation loss",
        }

    def log_metrics(self, loggers, outputs):
        for logger in loggers:
            logger(outputs)

    def compute_loggers(self, loggers, epoch, logger):
        for logger_ in loggers:
            logger_.compute(epoch, logger)

    def train_dataloader(self):
        dataloader = data.DataLoader(
            self.dataset_train,
            batch_size=self.training_args.batch_size,
            shuffle=True,
            num_workers=self.training_args.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )

        return dataloader

    def val_dataloader(self):
        dataloader = data.DataLoader(
            self.dataset_val,
            batch_size=self.training_args.batch_size,
            shuffle=True,
            num_workers=self.training_args.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )

        return dataloader
