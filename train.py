from argument_parser import parse_args
from data.dataset import get_dataset
from model.deblurrer_lightning_module import DeblurrerLightningModule
from model.modules.db_generator import DBGenerator
from model.modules.discriminator import Discriminator
import pytorch_lightning as pl
from clearml import Task
from loggers.loggers import setup_loggers


def _train(args):
    Task.force_requirements_env_freeze(False, "requirements.txt")

    task = Task.init(
        project_name="im2math",
        task_name=args.task_name,
        output_uri="gs://ai-experiments-artifacts",
        task_type=Task.TaskTypes.training,
    )

    tags = [str(args.dataset)]
    if args.tags:
        tags.extend(args.tags.split(","))
    task.add_tags(tags)

    if args.clearml_queue:
        task.set_base_docker(
            docker_cmd="nvidia/cuda:11.4.0-base-ubuntu20.04",
            docker_arguments=[
                "--shm-size 64g",
                "--memory 64g",
                "--rm",
            ],
        )
        task.execute_remotely(queue_name=args.clearml_queue)

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

    setup_loggers(args, module)

    trainer = pl.Trainer(
        log_every_n_steps=args.log_every_n_steps,
        gpus=int(args.ngpus) if str(args.ngpus).isnumeric() else args.ngpus,
        accelerator="dp",
    )
    trainer.fit(module, module.train_dataloader())


def main():
    args = parse_args()
    _train(args)


if __name__ == "__main__":
    main()
