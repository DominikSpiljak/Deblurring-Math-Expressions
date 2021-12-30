import logging

import torch
from torchvision.utils import make_grid


class ImageLogger:
    def __init__(self, max_logged_per_epoch, batch_size):
        self.max_logged_per_epoch = max_logged_per_epoch
        self.batch_size = batch_size
        self.images = []
        self.categories = ["blurred", "non_blurred", "deblurred"]

    def __call__(self, outputs):
        if (
            "optimizer_idx" not in outputs or torch.all(outputs["optimizer_idx"] == 0)
        ) and len(self.images) <= self.max_logged_per_epoch:
            self.images.append(
                torch.cat(
                    tuple(
                        outputs[category].unsqueeze(0)
                        for category in self.categories
                        if category in outputs
                    ),
                    1,
                )
            )

    def compute(self, epoch, logger):
        for index, image in enumerate(self.images):
            logging.info("Generating grid for image of shape %s", image.shape)
            grid = make_grid(
                image,
                normalize=True,
                nrow=1,
            )
            logger.experiment.add_images(f"Epoch {epoch}, batch {index}", grid, 0)
        self.images = []
