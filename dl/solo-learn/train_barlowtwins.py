import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDPPlugin

from solo.methods import BarlowTwins  # imports the method class
from solo.utils.checkpointer import Checkpointer

# some data utilities
# we need one dataloader to train an online linear classifier
# (don't worry, the rest of the model has no idea of this classifier, so it doesn't use label info)
from solo.utils.classification_dataloader import prepare_data as prepare_data_classification

# and some utilities to perform data loading for the method itself, including augmentation pipelines
from solo.utils.pretrain_dataloader import (
    prepare_dataloader,
    prepare_datasets,
    prepare_n_crop_transform,
    prepare_transform,
)

"""

https://solo-learn.readthedocs.io/en/latest/tutorials/overview.html
"""

# common parameters for all methods
# some parameters for extra functionally are missing, but don't mind this for now.
base_kwargs = {
    "backbone": "resnet18",
    "num_classes": 10,
    "cifar": True,
    "zero_init_residual": True,
    "max_epochs": 100,
    "optimizer": "sgd",
    "lars": True,
    "lr": 0.01,
    "gpus": "0",
    "grad_clip_lars": True,
    "weight_decay": 0.00001,
    "classifier_lr": 0.5,
    "exclude_bias_n_norm": True,
    "accumulate_grad_batches": 1,
    "extra_optimizer_args": {"momentum": 0.9},
    "scheduler": "warmup_cosine",
    "min_lr": 0.0,
    "warmup_start_lr": 0.0,
    "warmup_epochs": 10,
    "num_crops_per_aug": [2, 0],
    "num_large_crops": 2,
    "num_small_crops": 0,
    "eta_lars": 0.02,
    "lr_decay_steps": None,
    "dali_device": "gpu",
    "batch_size": 256,
    "num_workers": 4,
    "data_dir": "/data/datasets",
    "train_dir": "cifar10/train",
    "val_dir": "cifar10/val",
    "dataset": "cifar10",
    "name": "barlow-cifar10",
}

# barlow specific parameters
method_kwargs = {
    "proj_hidden_dim": 2048,
    "proj_output_dim": 2048,
    "lamb": 5e-3,
    "scale_loss": 0.025,
    "backbone_args": {"cifar": True, "zero_init_residual": True},
}

kwargs = {**base_kwargs, **method_kwargs}

model = BarlowTwins(**kwargs)

# we first prepare our single transformation pipeline
transform_kwargs = {
    "brightness": 0.4,
    "contrast": 0.4,
    "saturation": 0.2,
    "hue": 0.1,
    "gaussian_prob": 0.0,
    "solarization_prob": 0.0,
}
transform = [prepare_transform("cifar10", **transform_kwargs)]

# then, we wrap the pipepline using this utility function
# to make it produce an arbitrary number of crops
transform = prepare_n_crop_transform(transform, num_crops_per_aug=[2])

# finally, we produce the Dataset/Dataloader classes
train_dataset = prepare_datasets(
    "cifar10",
    transform,
    data_dir="./",
    train_dir=None,
    no_labels=False,
)
train_loader = prepare_dataloader(
    train_dataset, batch_size=base_kwargs["batch_size"], num_workers=base_kwargs["num_workers"]
)

# we will also create a validation dataloader to automatically
# check how well our models is doing in an online fashion.
_, val_loader = prepare_data_classification(
    "cifar10",
    data_dir="./",
    train_dir=None,
    val_dir=None,
    batch_size=base_kwargs["batch_size"],
    num_workers=base_kwargs["num_workers"],
)

wandb_logger = WandbLogger(
    name="barlow-cifar10",  # name of the experiment
    project="self-supervised",  # name of the wandb project
    entity=None,
    offline=False,
)
wandb_logger.watch(model, log="gradients", log_freq=100)

callbacks = []

# automatically log our learning rate
lr_monitor = LearningRateMonitor(logging_interval="epoch")
callbacks.append(lr_monitor)

# checkpointer can automatically log your parameters,
# but we need to wrap it on a Namespace object
from argparse import Namespace

args = Namespace(**kwargs)
# saves the checkout after every epoch
ckpt = Checkpointer(
    args,
    logdir="checkpoints/barlow",
    frequency=1,
)
callbacks.append(ckpt)

trainer = Trainer.from_argparse_args(
    args,
    logger=wandb_logger,
    callbacks=callbacks,
    plugins=DDPPlugin(find_unused_parameters=True),
    checkpoint_callback=False,
    terminate_on_nan=True,
    accelerator="ddp",
)


if __name__=="__main__":
# python3 main_pretrain.py \
#     --dataset cifar10 \
#     --backbone resnet18 \
#     --data_dir ./datasets \
#     --max_epochs 1000 \
#     --gpus 0 \
#     --num_workers 4 \
#     --precision 16 \
#     --optimizer sgd \
#     --lars \
#     --grad_clip_lars \
#     --eta_lars 0.02 \
#     --exclude_bias_n_norm \
#     --scheduler warmup_cosine \
#     --lr 0.3 \
#     --weight_decay 1e-4 \
#     --batch_size 256 \
#     --brightness 0.4 \
#     --contrast 0.4 \
#     --saturation 0.2 \
#     --hue 0.1 \
#     --gaussian_prob 0.0 \
#     --solarization_prob 0.0 \
#     --name barlow-cifar10 \
#     --project self-superivsed \
#     --wandb \
#     --save_checkpoint \
#     --method barlow_twins \
#     --proj_hidden_dim 2048 \
#     --output_dim 2048 \
#     --scale_loss 0.1
    trainer.fit(model, train_loader, val_loader)
