from argparse import ArgumentParser, Namespace
from pathlib import Path


class DotDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__  # type: ignore
    __delattr__ = dict.__delitem__  # type: ignore


def parse_args():
    parser = ArgumentParser(
        prog="Deblurring math expressions",
        description="Script that trains/evaluates/exploits model for deblurring",
    )

    data = parser.add_argument_group("data")
    training = parser.add_argument_group("training")
    model = parser.add_argument_group("model")
    clearml = parser.add_argument_group("clearml")
    logging = parser.add_argument_group("logging")

    data.add_argument(
        "--dataset",
        help="Path to dataset",
        type=Path,
        default=Path("/datasets/im2math/2021-11-07-dataset-full-quality/"),
    )

    data.add_argument(
        "--img-size", help="Image size", nargs=2, default=[512, 512], type=int
    )
    data.add_argument(
        "--sigmas",
        help="Sigmas for Gaussian blur",
        nargs=2,
        default=[4, 12],
        type=float,
    )
    data.add_argument(
        "--kernel-size",
        help="Kernel size for Gaussian blur",
        default=[17, 35],
        nargs=2,
        type=int,
    )

    model.add_argument(
        "--num-res-blocks",
        help="Number of ResNet blocks inside MIMOUnet block",
        default=8,
        type=int,
    )

    training.add_argument(
        "--learning-rate", help="Learning rate of the model", type=float, default=1e-3
    )
    training.add_argument(
        "--batch-size", help="Batch size for training", type=int, default=8
    )
    training.add_argument(
        "--alpha", help="Alpha parameter for loss calculating", type=float, default=0.1
    )
    training.add_argument(
        "--num-workers", help="Number of workers", type=int, default=8
    )
    training.add_argument(
        "--eval-mode",
        help="Enables evaluation mode for a checkpoint",
        action="store_true",
    )
    training.add_argument("--checkpoint", help="Bucket path to checkpoint")
    training.add_argument(
        "--limit-train-batches", help="Maximum train batches for an epoch", default=256
    )
    training.add_argument(
        "--limit-val-batches",
        help="Maximum validation batches for an epoch",
        default=64,
    )
    clearml.add_argument("--clearml-queue", help="ClearML queue used for training")
    clearml.add_argument(
        "--task-name",
        help="ClearML task name used for training",
        default="Image deblurring",
    )
    clearml.add_argument("--tags", help="Tags used for ClearML")
    clearml.add_argument(
        "--ngpus",
        help="Number of gpus to use. If 1 number is entered then"
        "that many gpus will be used, you can also enter numbers "
        "seperated by comma for specific gpu devices (1,2,3).",
        default=-1,
    )
    logging.add_argument(
        "--max-batches-logged-per-epoch",
        help="Number of image batches logged per epoch",
        type=int,
        default=2,
    )
    logging.add_argument(
        "--log-every-n-steps", help="Time interval for logging", type=int, default=10
    )
    logging.add_argument(
        "--disable-image-logging",
        help="Wether to disable model generated images logging",
        action="store_true",
    )
    logging.add_argument(
        "--save-top-k", help="Top k checkpoints to save", type=int, default=1
    )

    args = parser.parse_args()

    arg_groups = DotDict()

    for group in parser._action_groups:
        group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
        arg_groups[group.title] = Namespace(**group_dict)

    return arg_groups
