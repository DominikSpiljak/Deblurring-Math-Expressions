from argument_parser import parse_args
from data.dataset import get_dataset
from model.deblurrer_lightning_module import DeblurrerLightningModule
from model.modules.db_generator import DBGenerator
from model.modules.discriminator import Discriminator
import pytorch_lightning as pl


def _train(args):
    dataset_train, dataset_val = get_dataset(args.dataset, args.img_size)

    module = DeblurrerLightningModule(
        db_generator=DBGenerator(),
        discriminator=Discriminator(),
        dataset_train=dataset_train,
        dataset_val=dataset_val,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        alpha=args.alpha,
    )

    trainer = pl.Trainer(log_every_n_steps=args.log_every_n_steps)
    trainer.fit(module, module.train_dataloader())


def main():
    args = parse_args()
    _train(args)


if __name__ == "__main__":
    main()
