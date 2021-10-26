import torch
from torchvision.utils import make_grid


def setup_loggers(args, module):
    module.train_loggers = []
    module.validation_loggers = []

    if not args.disable_image_logging:
        module.validation_loggers.append(
            ImageLogger(args.max_batches_logged_per_epoch, args.batch_size)
        )


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
                    (outputs["blurred"], outputs["deblurred"]),
                    0,
                )
            )

    def compute(self, epoch, logger):
        for index, image in enumerate(self.images):
            grid = make_grid(
                image,
                normalize=True,
                nrow=self.batch_size,
            )
            logger.experiment.add_image(f"Epoch {epoch}, batch {index}", grid, 0)
        self.images = []
