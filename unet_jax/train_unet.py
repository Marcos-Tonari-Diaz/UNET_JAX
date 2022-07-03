from typing import Dict

import jax
import optax
import numpy as np

from data_loader import load_dataset, UnetTestDataGenerator, UnetTrainDataGenerator
from model import UnetJAX
from unet_training import UnetTrainState, train_epoch, eval_model, print_metrics, get_vmap_loss_grad, get_vmap_compute_metrics
from unet_utils import get_date_string

from torch.utils.tensorboard import SummaryWriter
import tensorflow as tf


def train_unet():
    input_img_size = 512
    learning_rate = 1e-2
    momentum = 0.99
    num_epochs = 4
    mini_batch_size = 4
    steps_per_epoch = 1
    train_split_size = 0.5
    rng_seed = 0
    datagen_seed = 1

    print(f'mini batch size: {mini_batch_size}')
    print(f'steps per epoch: {steps_per_epoch}')

    summary_writer = SummaryWriter("logs/"+get_date_string())

    paths = {"images": "../data/isbi2015/train/image/*.png",
             "masks": "../data/isbi2015/train/label/*.png"}
    dataset = load_dataset(paths, train_split_size=train_split_size)

    data_gen_args = dict(
        rotation_range=15.,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=50,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='constant'
    )
    train_datagen = UnetTrainDataGenerator(
        dataset['train'], data_gen_args, seed=datagen_seed, batch_size=mini_batch_size, steps_per_epoch=steps_per_epoch)

    test_datagen = UnetTestDataGenerator(
        dataset['test'], batch_size=mini_batch_size)

    unet = UnetJAX(input_image_size=input_img_size,
                   use_activation=False, use_padding=True)
    rng_key = jax.random.PRNGKey(rng_seed)
    unet_params = unet.init_params(rng_key)

    optimizer = optax.sgd(learning_rate=learning_rate, momentum=momentum)

    unet_train_state = UnetTrainState.create(
        apply_fn=unet.apply,
        params=unet_params,
        tx=optimizer,
        compute_loss_grad=get_vmap_loss_grad(),
        vmap_compute_metrics=get_vmap_compute_metrics()
    )

    # replicated_train_state = replicate(unet_train_state)

    for epoch in range(num_epochs):
        unet_train_state, train_metrics = train_epoch(
            unet_train_state, data_generator=train_datagen)
        print_metrics(train_metrics, epoch, "train epoch")

        test_metrics = eval_model(
            unet_train_state, data_generator=test_datagen)
        print_metrics(test_metrics, epoch, "test epoch")
        test_datagen = UnetTestDataGenerator(
            dataset['test'], batch_size=mini_batch_size)

        summary_writer.add_scalars(f'loss', {"train": float(np.array(train_metrics["loss"])),
                                             "test": float(np.array(test_metrics["loss"]))}, epoch)
        summary_writer.add_scalars(f'accuracy', {"train": float(np.array(train_metrics["accuracy"])),
                                                 "test": float(np.array(test_metrics["accuracy"]))}, epoch)

        # plot_predictions(dataset, unet, unet_train_state, epoch)

    summary_writer.close()


if __name__ == "__main__":
    tf.config.set_visible_devices([], device_type='GPU')
    train_unet()
