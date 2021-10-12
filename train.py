from argument_parser import parse_args
from data.dataset import get_dataset
from model.deblurrer_lightning_module import DeblurrerLightningModule
from model.modules.db_generator import DBGenerator
from model.modules.discriminator import Discriminator
import pytorch_lightning as pl


def _train(args):
    blurred_dataset = get_dataset(args.blurred_dataset, args.img_size, label="1")
    non_blurred_dataset = get_dataset(
        args.non_blurred_dataset, args.img_size, label="0"
    )

    module = DeblurrerLightningModule(
        db_generator=DBGenerator(),
        discriminator=Discriminator(),
        non_blurred_dataset=non_blurred_dataset,
        blurred_dataset=blurred_dataset,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    trainer = pl.Trainer(log_every_n_steps=args.log_every_n_steps)
    trainer.fit(module, module.train_dataloader())


def main():
    args = parse_args()
    _train(args)


if __name__ == "__main__":
    main()
