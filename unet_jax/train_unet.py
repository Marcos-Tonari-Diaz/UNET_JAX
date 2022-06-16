from threading import current_thread
from typing import Dict
import jax

import numpy as np
from model import UnetJAX
import optax

from data_loader import prepare_dataset, UnetTrainDataGenerator
from unet_training import UnetTrainState

from absl import logging
from torch.utils.tensorboard import SummaryWriter


from keras_unet_utils import plot_imgs, get_date_string

from datetime import datetime
import jax.numpy as jnp

def plot_predictions(dataset, unet, unet_train_state, epoch):
    pred_masks = []
    for test_img in dataset['test']['images']:
        mask_pred = unet.apply(
            {"params": unet_train_state.train_state.params}, test_img.reshape((1,)+test_img.shape))[0]
        mask_pred = jnp.round(jax.nn.sigmoid(mask_pred))
        pred_masks.append(mask_pred)
    pred_masks = np.array(pred_masks)
    original_imgs = np.array(dataset['test']['images'])
    original_masks = np.array(dataset['test']['masks'])
    # print(pred_imgs.shape)
    # print(original_imgs.shape)
    # print(original_masks.shape)

    plot_imgs(
        original_imgs,
        original_masks,
        pred_imgs=pred_masks,
        nm_img_to_plot=1,
        figsize=10,
        epoch=epoch)


def train_unet():
    learning_rate = 1e-2
    momentum = 0.99
    num_epochs = 10
    steps_per_epoch = 100
    train_split_size = 0.5

    summary_writer = SummaryWriter("logs/"+get_date_string())

    # dataset
    paths = {"images": "../data/isbi2015/train/image/*.png",
             "masks": "../data/isbi2015/train/label/*.png"}
    dataset = prepare_dataset(paths, train_split_size=train_split_size)
    unet_datagen = UnetTrainDataGenerator(
        dataset["train"]["images"], dataset["train"]["masks"], seed=1)

    unet = UnetJAX(input_image_size=512,
                   use_activation=False, use_padding=True)
    optimizer = optax.sgd(learning_rate=learning_rate, momentum=momentum)
    unet_train_state = UnetTrainState(
        unet, optimizer, seed=0, steps_per_epoch=steps_per_epoch)

    for epoch in range(num_epochs):
        train_metrics: Dict = unet_train_state.train_epoch(
            data_generator=unet_datagen)
        test_metrics: Dict = unet_train_state.eval_model(
            test_dataset=dataset["test"])

        summary_writer.add_scalars(f'loss', {"train": float(np.array(train_metrics["loss"])),
                                             "test": float(np.array(test_metrics["loss"]))},
                                   unet_train_state.current_epoch)

        summary_writer.add_scalars(f'accuracy', {"train": float(np.array(train_metrics["accuracy"])),
                                                 "test": float(np.array(test_metrics["accuracy"]))},
                                   unet_train_state.current_epoch)

        # summary_writer.add_scalar(f'train/loss', float(np.array(train_metrics["loss"])),
        #                           unet_train_state.current_epoch)
        # summary_writer.add_scalar(f'train/accuracy', float(np.array(train_metrics["accuracy"])),
        #                           unet_train_state.current_epoch)
        # summary_writer.add_scalar(f'test/loss', float(np.array(test_metrics["loss"])),
        #                           unet_train_state.current_epoch)
        # summary_writer.add_scalar(f'test/accuracy', float(np.array(test_metrics["accuracy"])),
        #                           unet_train_state.current_epoch)

    plot_predictions(dataset, unet, unet_train_state, epoch)
    summary_writer.close()


if __name__ == "__main__":
    train_unet()
