from typing import Dict
import jax

import numpy as np
from model import UnetJAX
import optax

from data_loader import prepare_dataset, UnetTrainDataGenerator
from unet_training_jit import UnetTrainState

from absl import logging
from torch.utils.tensorboard import SummaryWriter

from keras_unet_utils import plot_imgs, get_date_string

import jax.numpy as jnp


def plot_predictions(dataset, unet, unet_train_state, epoch):
    pred_logits = []
    pred_masks = []
    for test_img in dataset['test']['images']:
        logits_pred = unet.apply(
            {"params": unet_train_state.train_state.params}, test_img.reshape((1,)+test_img.shape))[0]
        mask_pred = jnp.round(jax.nn.sigmoid(logits_pred))
        pred_logits.append(logits_pred)
        pred_masks.append(mask_pred)
    pred_masks = np.array(pred_masks)
    pred_logits = np.array(pred_logits)
    original_imgs = np.array(dataset['test']['images'])
    original_masks = np.array(dataset['test']['masks'])

    plot_imgs(
        original_imgs,
        original_masks,
        pred_logits=pred_logits,
        pred_masks=pred_masks,
        nm_img_to_plot=1,
        figsize=10,
        # save_path="experiment-"+get_date_string()+"_epoch-" + str(epoch)+"_prediction.png")
        save_path=f"experiment-{get_date_string()}_epoch-{epoch}_prediction.png")


def train_unet():
    learning_rate = 1e-2
    momentum = 0.99
    num_epochs = 1
    steps_per_epoch = 1
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

    # UnetTrainState.registerAsPyTree()
    unet_train_state = UnetTrainState(steps_per_epoch=steps_per_epoch)

    train_state = unet_train_state.create_training_state(unet, optimizer)

    for epoch in range(num_epochs):
        train_metrics: Dict = unet_train_state.train_epoch(train_state,
                                                           data_generator=unet_datagen)
        test_metrics: Dict = unet_train_state.eval_model(train_state,
                                                         test_dataset=dataset["test"])

        summary_writer.add_scalars(f'loss', {"train": float(np.array(train_metrics["loss"])),
                                             "test": float(np.array(test_metrics["loss"]))},
                                   unet_train_state.current_epoch)

        summary_writer.add_scalars(f'accuracy', {"train": float(np.array(train_metrics["accuracy"])),
                                                 "test": float(np.array(test_metrics["accuracy"]))},
                                   unet_train_state.current_epoch)

        plot_predictions(dataset, unet, unet_train_state, epoch)

    summary_writer.close()


if __name__ == "__main__":
    train_unet()
