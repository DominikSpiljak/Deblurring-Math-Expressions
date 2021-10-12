from argparse import ArgumentParser
from pathlib import Path


def parse_args():
    parser = ArgumentParser(
        prog="Deblurring math expressions",
        description="Script that trains/evaluates/exploits model for deblurring",
    )

    parser.add_argument("--blurred-dataset", help="Path to blurred dataset", type=Path)
    parser.add_argument(
        "--non-blurred-dataset", help="Path to non-blurred dataset", type=Path
    )

    parser.add_argument("--img-size", help="Image size", nargs=2, default=[128, 128])

    parser.add_argument(
        "--learning-rate", help="Learning rate of the model", type=float, default=1e-4
    )
    parser.add_argument(
        "--batch-size", help="Batch size for training", type=int, default=4
    )
    parser.add_argument(
        "--log-every-n-steps", help="Time interval for logging", type=int, default=100
    )
    parser.add_argument("--num-workers", help="Number of workers", type=int, default=1)

    return parser.parse_args()
