import logging

import pytorch_lightning as pl
from clearml import StorageManager, Task
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from argument_parser import parse_args
from data.dataset import get_dataset
from loggers.loggers import setup_loggers
from model.mimo_lightning_module import MIMOUnetModule
from model.mimo_unet_modules.mimo_unet import MIMOUnet


def _train(args):
    Task.force_requirements_env_freeze(False, "requirements.txt")

    task = Task.init(
        project_name="im2math",
        task_name=args.task_name,
        output_uri="gs://ai-experiments-artifacts",
        task_type=Task.TaskTypes.testing if args.eval_mode else Task.TaskTypes.training,
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

    dataset_train, dataset_val = get_dataset(
        args.dataset, args.img_size, args.kernel_size, args.sigmas
    )

    module = MIMOUnetModule(
        mimo_unet=MIMOUnet(),
        dataset_train=dataset_train,
        dataset_val=dataset_val,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        alpha=args.alpha,
    )
    if args.checkpoint:
        if str(args.checkpoint).startswith("gs"):
            loaded_model = StorageManager.get_local_copy(args.checkpoint)
        else:
            loaded_model = args.checkpoint

        module = MIMOUnetModule.load_from_checkpoint(loaded_model)

    setup_loggers(args, module)

    metric_monitor_callbacks = [
        ModelCheckpoint(
            save_last=True,
            verbose=True,
            monitor="Validation loss",
            save_top_k=args.save_top_k,
            mode="min",
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    trainer = pl.Trainer(
        callbacks=metric_monitor_callbacks,
        log_every_n_steps=args.log_every_n_steps,
        gpus=int(args.ngpus) if str(args.ngpus).isnumeric() else args.ngpus,
        accelerator="ddp",
    )

    if not args.eval_mode:
        trainer.fit(module)

    trainer.validate(module)


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    _train(args)


if __name__ == "__main__":
    main()
