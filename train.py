from argument_parser import parse_args
from data.dataset import get_dataset


def _train(args):
    blurred_dataset = get_dataset(args.blurred_dataset, args.img_size, label="1")
    non_blurred_dataset = get_dataset(
        args.non_blurred_dataset, args.img_size, label="0"
    )


def main():
    args = parse_args()
    _train(args)


if __name__ == "__main__":
    main()
