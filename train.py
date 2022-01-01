import logging

import pytorch_lightning as pl
from clearml import StorageManager, Task
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from argument_parser import parse_args
from model.mimo_lightning_module import MIMOUnetModule


def _train(args, pl_module):
    Task.force_requirements_env_freeze(False, "requirements.txt")

    task = Task.init(
        project_name="im2math",
        task_name=args.clearml.task_name,
        output_uri="gs://ai-experiments-artifacts",
        task_type=Task.TaskTypes.testing
        if args.training.eval_mode
        else Task.TaskTypes.training,
    )

    tags = [str(args.data.dataset)]
    if args.clearml.tags:
        tags.extend(args.clearml.tags.split(","))
    task.add_tags(tags)

    if args.clearml.clearml_queue:
        task.set_base_docker(
            docker_cmd="nvidia/cuda:11.4.0-base-ubuntu20.04",
            docker_arguments=[
                "--shm-size 64g",
                "--memory 64g",
                "--rm",
            ],
        )
        task.execute_remotely(queue_name=args.clearml.clearml_queue)

    module = pl_module(
        data_args=args.data,
        model_args=args.model,
        training_args=args.training,
        logger_args=args.logging,
    )
    if args.training.checkpoint:
        if str(args.training.checkpoint).startswith("gs"):
            model_checkpoint = StorageManager.get_local_copy(args.training.checkpoint)
        else:
            model_checkpoint = args.training.checkpoint

        module = pl_module.load_from_checkpoint(
            model_checkpoint,
            data_args=args.data,
            training_args=args.training,
            logger_args=args.logging,
            loaded_from_checkpoint=True,
        )

    metric_monitor_callbacks = [
        ModelCheckpoint(
            save_last=True,
            verbose=True,
            monitor="Validation loss",
            save_top_k=args.logging.save_top_k,
            mode="min",
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    trainer = pl.Trainer(
        callbacks=metric_monitor_callbacks,
        log_every_n_steps=args.logging.log_every_n_steps,
        gpus=int(args.clearml.ngpus)
        if str(args.clearml.ngpus).isnumeric()
        else args.clearml.ngpus,
        strategy="dp",
        limit_train_batches=args.training.limit_train_batches,
        limit_val_batches=args.training.limit_val_batches,
    )

    if not args.training.eval_mode:
        trainer.fit(module)

    trainer.validate(module)


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    _train(args, pl_module=MIMOUnetModule)


if __name__ == "__main__":
    main()
