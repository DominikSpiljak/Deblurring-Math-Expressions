import logging
from argparse import Namespace

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils import data

from data.dataset import get_dataset_blur
from loggers.loggers import ImageLogger
from model.bgan_modules.blur_generator import BGenerator
from model.bgan_modules.discriminator import Discriminator, ResNet18Discriminator


def calculate_bce_loss(predictions, real):
    return F.binary_cross_entropy_with_logits(predictions, real)


def calculate_l1_loss(generated, ground_truth):
    return F.l1_loss(generated, ground_truth)


def collate_fn(batch):
    non_blurred = torch.stack([img[0] for img in batch])
    blurred = torch.stack([img[1] for img in batch])

    return non_blurred, blurred


class RealisticBlurrerModule(pl.LightningModule):
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
        self.g_model = BGenerator(
            **{k: v for k, v in vars(self.model_args).items() if v is not None}
        )
        self.d_model = ResNet18Discriminator()
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
                    ["non_blurred", "blurred"],
                    "Validation",
                )
            )

    def setup_datasets(self):
        return get_dataset_blur(
            blurred_dataset_path=self.data_args.dataset_blurred,
            non_blurred_dataset_path=self.data_args.dataset,
            img_size=self.data_args.img_size,
        )

    def generator_forward(self, batch):
        noise = torch.randn((batch.shape[0], 1, *batch.shape[2:]), device=self.device)
        g_input = torch.cat((batch, noise), dim=1)
        return self.g_model(g_input)

    def training_step(self, batch, batch_idx, optimizer_idx):
        non_blurred_images, blurred_images = batch
        if optimizer_idx == 0:
            real_labels = torch.ones(non_blurred_images.size(0), 1, device=self.device)
            blurred = self.generator_forward(non_blurred_images)

            prediction = self.d_model(blurred)
            g_loss_bce = calculate_bce_loss(prediction, real_labels)
            g_loss_l1 = calculate_l1_loss(blurred, non_blurred_images)

            g_loss = g_loss_l1 + g_loss_bce * self.training_args.alpha
            self.log("Generator loss", g_loss, prog_bar=True)

            return {
                "loss": g_loss,
                "non_blurred": non_blurred_images,
                "blurred": blurred.detach(),
                "optimizer_idx": optimizer_idx,
            }

        if optimizer_idx == 1:
            fake_labels = torch.zeros(blurred_images.size(0), 1, device=self.device)
            real_labels = torch.ones(blurred_images.size(0), 1, device=self.device)

            prediction = self.d_model(blurred_images)
            d_real_loss = calculate_bce_loss(prediction, real_labels)

            prediction = self.d_model(
                self.generator_forward(non_blurred_images).detach()
            )
            d_fake_loss = calculate_bce_loss(prediction, fake_labels)

            d_loss = (d_real_loss + d_fake_loss) / 2.0

            self.log("Discriminator loss", d_loss, prog_bar=True)

            return {"loss": d_loss, "optimizer_idx": optimizer_idx}

    def validation_step(self, batch, batch_idx):
        real_labels = torch.ones(batch.size(0), 1, device=self.device)
        blurred = self.generator_forward(batch)

        prediction = self.d_model(blurred)
        g_loss_bce = calculate_bce_loss(prediction, real_labels)
        g_loss_l1 = calculate_l1_loss(blurred, batch)

        g_loss = g_loss_l1 + g_loss_bce * self.training_args.alpha
        self.log("Generator loss", g_loss, prog_bar=True)

        return {"loss": g_loss, "non_blurred": batch, "blurred": blurred.detach()}

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
        g_optimizer = torch.optim.Adam(
            self.g_model.parameters(), lr=self.training_args.learning_rate
        )
        d_optimizer = torch.optim.Adam(
            self.d_model.parameters(), lr=self.training_args.learning_rate
        )

        g_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            g_optimizer, mode="min", factor=0.9, patience=3, min_lr=1e-5
        )
        d_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            g_optimizer, mode="min", factor=0.9, patience=3, min_lr=1e-5
        )
        return (
            {
                "optimizer": g_optimizer,
                "lr_scheduler": {
                    "scheduler": g_scheduler,
                    "monitor": "Validation loss",
                },
            },
            {
                "optimizer": d_optimizer,
                "lr_scheduler": {
                    "scheduler": d_scheduler,
                    "monitor": "Validation loss",
                },
            },
        )

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
        )

        return dataloader
