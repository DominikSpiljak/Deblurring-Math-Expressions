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

    parser.add_argument("--img-size", help="Image size", nargs=2, default=[512, 512])

    return parser.parse_args()
