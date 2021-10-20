import torch
from torchvision.utils import make_grid


def setup_loggers(args, module):
    module.train_loggers = []

    if not args.disable_image_logging:
        module.train_loggers.append(
            ImageLogger(args.max_batches_logged_per_epoch, logger=module.logger)
        )


class ImageLogger:
    def __init__(self, max_logged_per_epoch, logger):
        self.max_logged_per_epoch = max_logged_per_epoch
        self.images = []
        self.logger = logger

    def __call__(self, outputs):
        if (
            torch.all(outputs["optimizer_idx"] == 0)
            and len(self.images) <= self.max_logged_per_epoch
        ):
            self.images.append(torch.cat((outputs["blurred"], outputs["deblurred"]), 0))

    def compute(self, epoch):
        for index, image in enumerate(self.images):
            grid = make_grid(
                image,
                normalize=True,
            )
            self.logger.experiment.add_image(f"Epoch {epoch}, batch {index}", grid, 0)
