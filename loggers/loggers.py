import logging

import torch
from torchvision.utils import make_grid


class ImageLogger:
    def __init__(self, max_logged_per_epoch, batch_size):
        self.max_logged_per_epoch = max_logged_per_epoch
        self.batch_size = batch_size
        self.images = []

    def __call__(self, outputs):
        if (
            "optimizer_idx" not in outputs or torch.all(outputs["optimizer_idx"] == 0)
        ) and len(self.images) <= self.max_logged_per_epoch:
            self.images.append(
                torch.cat(
                    (
                        outputs["blurred"].unsqueeze(0),
                        outputs["deblurred"].unsqueeze(0),
                        outputs["non_blurred"].unsqueeze(0),
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
