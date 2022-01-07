import logging
from argument_parser import parse_args
from train import _train
from model.blurrer_lightning_module import RealisticBlurrerModule


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    _train(
        args,
        pl_module=RealisticBlurrerModule,
        checkpoint=args.training.blurrer_checkpoint,
    )


if __name__ == "__main__":
    main()
